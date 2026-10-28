import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding='utf-8')


def discover(root, out):
    """Schema discovery, not a hard-coded upload path; never modifies originals."""
    files = [p for p in root.rglob('*') if p.is_file() and
             not any(k in p.relative_to(root).parts for k in ('tmp', '.git', 'study', 'data', '__pycache__'))]
    groups = {'wide': [], 'fine': [], 'dcr': [], 'low': []}
    for p in files:
        if p.suffix.lower() != '.csv':
            continue
        rows = read_csv(p)
        if not rows:
            continue
        if 'DCR_ohm' in rows[0]:
            groups['dcr'].append(p)
        elif all(k in rows[0] for k in ('frequency_Hz','voltage_setpoint_Vrms','Rs_ohm','Ls_H','R_ohm','X_ohm')):
            v = float(rows[0]['voltage_setpoint_Vrms'])
            key = 'low' if v < .01 else ('fine' if 'R_direct_ohm' in rows[0] else 'wide')
            if key == 'low' or len(rows) >= 90:
                groups[key].append(p)
    chosen = {}
    for k, paths in groups.items():
        if paths:
            chosen[k] = sorted(paths, key=lambda p: (len(read_csv(p)), p.name))[-1]
    if not all(k in chosen for k in ('wide', 'fine', 'dcr')):
        raise FileNotFoundError(f'Required measured data missing: {groups}')
    extra = [p for p in files if p.suffix.lower() in ('.pdf', '.dlog') or '说明' in p.name
             or p.name in ('feasibility.md', 'monitor_panel_crosscheck.json','task_20261028.txt')]
    manifest = []
    raw = root/'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    for p in sorted(set(chosen.values()) | set(extra)):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        target = raw/(digest[:12]+'_'+p.name)
        if not target.exists():
            shutil.copy2(p, target)
        manifest.append(dict(path=str(p.relative_to(root)), sha256=digest, bytes=p.stat().st_size,
                             preserved_copy=str(target.relative_to(root))))
    dump(out/'input_manifest.json', manifest)
    missing = ['01_测量结果说明.txt', '04_结果说明与列定义.txt']
    dump(out/'discovery.json', dict(selected=chosen, candidates=groups,
         dlog=[str(p) for p in extra if p.suffix.lower()=='.dlog'],
         missing=[s for s in missing if not any(p.name==s for p in files)]))
    return chosen, [p for p in extra if p.suffix.lower()=='.dlog']


def observations(path):
    rows = read_csv(path)
    f = np.array([float(r['frequency_Hz']) for r in rows])
    z = np.array([complex(float(r.get('R_direct_ohm') or r['R_ohm']),
                         float(r.get('X_direct_ohm') or r['X_ohm'])) for r in rows])
    alt = np.array([complex(float(r['Rs_ohm']), 2*np.pi*float(r['frequency_Hz'])*float(r['Ls_H'])) for r in rows])
    polar = np.array([float(r['Z_ohm'])*np.exp(1j*np.deg2rad(float(r['phase_deg']))) for r in rows])
    unique = np.unique(f)
    return dict(f=unique, z=np.array([z[f==v].mean() for v in unique]), raw_rows=rows,
                duplicate_points=len(f)-len(unique), rx_crosscheck_ohm=float(np.max(abs(z-alt))),
                polar_crosscheck_ohm=float(np.max(abs(z-polar))))


def inspect_dlog(paths, out):
    """Unknown binary formats are never decoded by guessing a current scale."""
    records = []
    for p in paths:
        b = p.read_bytes()
        record = dict(path=str(p), bytes=len(b), prefix_hex=b[:128].hex(),
                      status='unparsed_binary_format', current_interpretation='instrument_port_only')
        # Accept an explicitly exported text file only if column identities are present.
        try:
            text = b.decode('utf-8-sig')
            if text.splitlines()[0].strip() == 't,V_port,I_port':
                rows = list(csv.DictReader(text.splitlines()))
                a = np.array([[float(r[k]) for k in ('t','V_port','I_port')] for r in rows])
                if np.all(np.diff(a[:,0])>0):
                    write_csv(out/(p.stem+'_port.csv'), rows)
                    record.update(status='parsed_named_text_export', samples=len(a),
                                  sample_rate_Hz=1/np.median(np.diff(a[:,0])))
        except (UnicodeError, ValueError, IndexError):
            pass
        records.append(record)
    dump(out/'dlog_status.json', dict(files=records, status='missing' if not paths else 'see_files',
         note='No coil-current calibration. Port current may include analyzer output capacitance; 25.4uF not identified.'))
