"""Wheel aggregate licenses must survive duplicate child LICENSE basenames."""
import importlib.util
from pathlib import Path


def builder():
    path=Path(__file__).resolve().parents[1]/'scripts/build_calibrated_candidate.py'
    spec=importlib.util.spec_from_file_location('public_builder_licenses',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_aggregate_and_conflicting_child_notices_survive_repeat_collection(tmp_path):
    entries={'example-1.0.dist-info/LICENSE.txt':b'wheel aggregate with bundled dependencies\r\n',
             'example/vendor/a/LICENSE.txt':b'first child\n',
             'example/vendor/b/LICENSE.txt':b'second child\n'}
    for name,data in entries.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    class Distribution:
        files=[Path(name) for name in entries]
        def locate_file(self,item):return tmp_path/item
    notices=tmp_path/'notices';notices.mkdir();b=builder()
    for _ in range(2):b.collect_distribution_notices('example',Distribution(),notices)
    assert (notices/'example_WHEEL_LICENSE.txt').read_bytes()==entries['example-1.0.dist-info/LICENSE.txt']
    assert sorted(path.read_bytes() for path in notices.iterdir())==sorted(entries.values())


def test_dist_info_licenses_layout_is_aggregate(tmp_path):
    item=Path('example-1.0.dist-info/licenses/LICENSE.txt')
    source=tmp_path/item;source.parent.mkdir(parents=True);source.write_bytes(b'aggregate')
    class Distribution:
        files=[item]
        def locate_file(self,item):return tmp_path/item
    notices=tmp_path/'notices';notices.mkdir()
    builder().collect_distribution_notices('example',Distribution(),notices)
    assert (notices/'example_WHEEL_LICENSE.txt').read_bytes()==b'aggregate'
