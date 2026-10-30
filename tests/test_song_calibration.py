import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from uart_buzzer.desktop.cache import LibraryAudioCache
from uart_buzzer.desktop.library import LibraryStore, inspect_paths
from uart_buzzer.desktop.song_calibration import (
    conversion_context, digest, file_hash, freeze_recording_context, legacy_auto_fit,
    profile_hash, resolve_profile, validate_recording,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def songs(tmp_path):
    profile = json.loads((ROOT / 'desktop_assets/music_profile.json').read_text())
    paths = []
    for name, frequency in [('A', 440), ('B', 660)]:
        path = tmp_path / (name + '.wav')
        x = (.15 * np.sin(2 * np.pi * frequency * np.arange(48000) / 16000)).astype(np.float32)
        wavfile.write(path, 16000, x)
        paths.append(path)
    store = LibraryStore(tmp_path / 'library.json')
    ids = store.add(inspect_paths(paths))
    model = ROOT / 'desktop_assets/impedance_fit.json'
    context = conversion_context(ids[0], str(paths[0]), file_hash(paths[0]), profile, file_hash(model))
    context.update(recording_id='frozen-A', actual_profile=copy.deepcopy(profile),
                   actual_profile_sha256=digest(profile), source_PCM_sha256='frozen-rendered-A')
    fitted = copy.deepcopy(profile)
    fitted['gains_db'][4] += 1.
    return store, ids, paths, profile, context, fitted, model


def test_song_only_persistence_duplicate_content_and_relink(songs):
    store, ids, paths, base, context, fitted, model = songs
    clone = paths[0].with_name('A-copy.wav')
    clone.write_bytes(paths[0].read_bytes())
    clone_id = store.add(inspect_paths([clone]))[0]
    store.bind_calibration(context, fitted, 'recording-A.wav')
    state=copy.deepcopy(store.data);state['tracks'][ids[0]]['calibration_enabled']=False;store.commit(state)
    store.add(inspect_paths([paths[0]]))
    assert store.data['tracks'][ids[0]]['calibration_enabled'] is False
    restarted = LibraryStore(store.path)
    record = restarted.data['tracks'][ids[0]]['calibration']
    active = resolve_profile(base, record, file_hash(paths[0]), 'music', file_hash(model))
    assert active['gains_db'] == fitted['gains_db'] and base['gains_db'] != fitted['gains_db']
    assert 'calibration' not in restarted.data['tracks'][ids[1]]
    assert 'calibration' not in restarted.data['tracks'][clone_id]
    moved = paths[0].with_name('renamed-A.wav')
    paths[0].rename(moved)
    new = restarted.relink(ids[0], moved)
    assert restarted.calibration_target(context) == new
    assert restarted.data['tracks'][new]['calibration'] == record
    assert restarted.data['tracks'][new]['calibration_enabled'] is False
    assert resolve_profile(base, record, file_hash(moved), 'music', file_hash(model)) == active
    restarted.bind_calibration(context, fitted, 'same-frozen-recording-after-move.wav')


def test_changed_content_same_stat_and_removed_song_cannot_receive_fit(songs):
    store, ids, paths, base, context, fitted, model = songs
    stat = paths[0].stat()
    raw = bytearray(paths[0].read_bytes()); raw[-8] ^= 1; paths[0].write_bytes(raw)
    import os
    os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='内容已改变'):
        store.bind_calibration(context, fitted, 'old-doc-recording.wav')
    store.remove([ids[0]], 'all')
    with pytest.raises(ValueError, match='已移除'):
        store.bind_calibration(context, fitted, 'old-doc-recording.wav')
    assert ids[0] not in store.data['tracks']


def test_overlay_never_resets_controls_and_new_fit_invalidates_only_its_cache(songs):
    store, ids, paths, base, context, fitted, model = songs
    cache = LibraryAudioCache(store.path.parent / 'prepared')
    original = cache.convert(paths[0], base)
    other = cache.convert(paths[1], base)
    store.bind_calibration(context, fitted, 'A1.wav')
    record = store.data['tracks'][ids[0]]['calibration']
    enabled = resolve_profile(base, record, context['source_sha256'], 'music', file_hash(model))
    first = cache.convert(paths[0], enabled)
    assert first.metadata['cache']['key'] != original.metadata['cache']['key']
    assert enabled['baud'] == base['baud'] and enabled['encoder'] == base['encoder']
    for field, value in [('baud', 921600), ('encoder', 'frame_aware_sigma_delta'), ('loudness_target_rms', .32)]:
        changed = copy.deepcopy(base); changed[field] = value
        assert '_song_calibration' not in resolve_profile(changed, record, context['source_sha256'], 'music', file_hash(model))
    assert '_song_calibration' not in resolve_profile(base, record, 'different-content', 'music', file_hash(model))
    assert '_song_calibration' not in resolve_profile(base, record, context['source_sha256'], 'electrical', file_hash(model))
    bypass = resolve_profile(base, record, context['source_sha256'], 'music', file_hash(model), False)
    with patch('uart_buzzer.desktop.cache.convert', side_effect=AssertionError('re-encoding baseline')):
        assert LibraryAudioCache(cache.folder).convert(paths[0], bypass).metadata['cache']['tier'] == 'disk'
        assert LibraryAudioCache(cache.folder).convert(paths[1], base).metadata['cache']['key'] == other.metadata['cache']['key']
    store.bind_calibration(context, fitted, 'A2-same-numeric-fit.wav')
    second = resolve_profile(base, store.data['tracks'][ids[0]]['calibration'], context['source_sha256'], 'music', file_hash(model))
    assert cache.key(paths[0], second, 'music', model, None, None) != first.metadata['cache']['key']
    assert len(list(cache.folder.glob('*.npz'))) == 3


def test_failed_library_commit_does_not_install_fit(songs):
    store, ids, paths, base, context, fitted, model = songs
    old = copy.deepcopy(store.data)
    with patch.object(store, '_atomic_write', side_effect=OSError('disk full')):
        with pytest.raises(OSError):store.bind_calibration(context, fitted, 'A.wav')
    assert store.data == old
    assert LibraryStore(store.path).data == old


def test_old_automatic_fit_detection_preserves_explicit_manual_intent_boundary():
    assert legacy_auto_fit({'name': '本机音乐录音拟合'}, 'music_fit_20261029_054334.json')
    assert legacy_auto_fit({'status': 'user_recording_fit_needs_other_music_validation'}, 'renamed.json')
    assert not legacy_auto_fit({'name': '用户手动均衡'}, 'manual.json')


def test_session_context_and_actual_profile_are_frozen_and_mismatch_is_rejected(songs):
    store, ids, paths, base, context, fitted, model = songs
    from types import SimpleNamespace
    doc = SimpleNamespace(profile=copy.deepcopy(base), metadata=dict(song_context=copy.deepcopy(context), source_PCM_sha256='rendered'))
    frozen = freeze_recording_context(doc)
    doc.profile['gains_db'][0] += 1
    assert frozen['actual_profile']['gains_db'] == base['gains_db']
    folder = store.path.parent / 'session';folder.mkdir()
    reference = folder / 'reference_processed.wav';capture = folder / 'microphone_raw.wav'
    reference.write_bytes(b'captured-reference');capture.write_bytes(b'captured-microphone')
    meta = dict(calibration_context=frozen, profile=frozen['actual_profile'], source_path=frozen['source_path'],
                completed=True, paused=False, stopped=False, error=None,
                calibration_reference_sha256=file_hash(reference), calibration_capture_sha256=file_hash(capture))
    rec = dict(calibration_context=copy.deepcopy(frozen), clipped_fraction=0., portaudio_status=[])
    (folder / 'playback.json').write_text(json.dumps(meta));(folder / 'recording.json').write_text(json.dumps(rec))
    assert validate_recording(folder, capture)[0] == frozen
    rec['calibration_context']['track_id'] = ids[1]
    (folder / 'recording.json').write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='来源不一致'):validate_recording(folder, capture)
    rec['calibration_context'] = frozen;(folder / 'recording.json').write_text(json.dumps(rec))
    capture.write_bytes(b'replaced-capture')
    with pytest.raises(ValueError, match='已改变'):validate_recording(folder, capture)


def test_legacy_five_component_cache_is_not_misrepresented_as_compatible(songs):
    store, ids, paths, base, context, fitted, model = songs
    cache = LibraryAudioCache(store.path.parent / 'prepared')
    from uart_buzzer.desktop import cache as C
    old = {k:v for k,v in C.PROCESSOR.items() if k not in ['beam.py','beam_model.py','_beam_kernels.py']}
    old['engine.py'] = 'actual-v131-different-DSP-fingerprint'
    with patch.dict(C.__dict__, PROCESSOR=old):legacy = cache.convert(paths[0], base)
    legacy_file = cache.folder / (legacy.metadata['cache']['key'] + '.npz')
    new = LibraryAudioCache(cache.folder).convert(paths[0], base)
    assert not new.metadata['cache']['hit'] and legacy_file.exists()


@pytest.mark.parametrize('restore_inside_conversion',[False,True])
def test_hash_decode_replacement_never_commits_wrong_source(songs,restore_inside_conversion):
    store, ids, paths, base, context, fitted, model = songs
    from uart_buzzer.desktop import cache as C
    from uart_buzzer.music import read_audio
    cache=LibraryAudioCache(store.path.parent/'prepared')
    original=paths[0].read_bytes();expected=read_audio(paths[0])[0].astype(np.float32)
    actual=C.convert
    def race(*args,**kwargs):
        paths[0].write_bytes(paths[1].read_bytes())
        result=actual(*args,**kwargs)
        if restore_inside_conversion:paths[0].write_bytes(original)
        return result
    with patch.object(C,'convert',side_effect=race):
        if restore_inside_conversion:
            doc=cache.convert(paths[0],base)
            np.testing.assert_array_equal(doc.source,expected)
        else:
            with pytest.raises(ValueError,match='文件已改变'):cache.convert(paths[0],base)
            assert not cache.memory and not list(cache.folder.glob('*.npz'))
    paths[0].write_bytes(original)
    for instance in [cache,LibraryAudioCache(cache.folder)]:
        np.testing.assert_array_equal(instance.convert(paths[0],base).source,expected)


@pytest.mark.parametrize('tier',['memory','disk'])
def test_unverified_legacy_poison_is_rejected_but_correct_legacy_is_reused(songs,tier):
    store, ids, paths, base, context, fitted, model = songs
    from uart_buzzer.desktop import cache as C
    cache=LibraryAudioCache(store.path.parent/'prepared')
    key=cache.key(paths[0],base,'music',None,None,None)
    wrong=C.convert(paths[1],base);wrong.metadata.pop('source_commit',None)
    cache.save(key,wrong)
    if tier=='memory':cache.remember(key,wrong)
    doc=cache.convert(paths[0],base)
    assert not doc.metadata['cache']['hit'] and doc.metadata['source_PCM_sha256']!=wrong.metadata['source_PCM_sha256']
    legacy=copy.deepcopy(doc);legacy.metadata.pop('source_commit',None);cache.memory.clear();cache.used=0;cache.save(key,legacy)
    with patch.object(C,'convert',side_effect=AssertionError('correct old UART must be reused')):
        reused=LibraryAudioCache(cache.folder).convert(paths[0],base)
    assert reused.metadata['cache']['tier']=='disk'
    assert reused.metadata['source_commit']['source_sha256']==file_hash(paths[0])
