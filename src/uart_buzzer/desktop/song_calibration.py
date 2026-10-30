"""Content-bound song calibration and immutable recording provenance.

Automatic fits never become a global profile. No DSP/model implementation lives here.
"""
import copy
import hashlib
import json
from pathlib import Path
import uuid

SCHEMA = 'uart-song-calibration-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path, cancel_check=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            if cancel_check and cancel_check():
                raise InterruptedError('操作已取消')
            h.update(chunk)
    return h.hexdigest()


def profile_hash(profile):
    ignored = {'name', 'status', 'evidence', 'clock_scale_estimate', '_song_calibration'}
    return digest({k: v for k, v in profile.items() if k not in ignored})


def legacy_auto_fit(profile, path):
    return (profile.get('status') in ('user_recording_fit_needs_other_music_validation', 'song_recording_fit')
            or profile.get('name') == '本机音乐录音拟合'
            or Path(path).stem.startswith('music_fit_'))


def resolve_profile(base, record, source_sha256, mode, model_sha256, enabled=True):
    result = copy.deepcopy(base)
    result.pop('_song_calibration', None)
    if not enabled or mode != 'music' or not record:
        return result
    if (record.get('schema') != SCHEMA or record.get('source_sha256') != source_sha256
            or record.get('base_profile_sha256') != profile_hash(base)
            or record.get('model_sha256') != model_sha256):
        return result
    fitted = record['profile']
    if digest(fitted) != record['profile_sha256']:
        raise ValueError('本曲校准配置校验失败，请重新拟合')
    result.update(gains_db=copy.deepcopy(fitted['gains_db']), name='本曲录音校准',
                  status='song_recording_fit', clock_scale_estimate=fitted.get('clock_scale_estimate'))
    result['_song_calibration'] = dict(id=record['id'], target_source_sha256=source_sha256,
                                       fitted_profile_sha256=record['profile_sha256'])
    return result


def conversion_context(track_id, path, source_sha256, base, model_sha256):
    return dict(schema=SCHEMA, track_id=track_id, source_path=str(path),
                source_sha256=source_sha256, base_profile_sha256=profile_hash(base),
                model_sha256=model_sha256)


def freeze_recording_context(document):
    context = copy.deepcopy(document.metadata.get('song_context'))
    if not context or not context.get('track_id') or not document.metadata.get('source_PCM_sha256'):
        return None
    context.update(recording_id=uuid.uuid4().hex, actual_profile=copy.deepcopy(document.profile),
                   actual_profile_sha256=digest(document.profile),
                   source_PCM_sha256=document.metadata['source_PCM_sha256'])
    return context


def validate_recording(folder, capture_path):
    folder = Path(folder)
    meta = json.loads((folder / 'playback.json').read_text(encoding='utf-8'))
    recording = json.loads((folder / 'recording.json').read_text(encoding='utf-8'))
    context = meta.get('calibration_context')
    if not context or context.get('schema') != SCHEMA or not context.get('track_id'):
        raise ValueError('旧录音缺少可证明的歌曲来源，请重新同步录音；不会猜测绑定歌曲')
    if recording.get('calibration_context') != context:
        raise ValueError('录音与播放的歌曲来源不一致')
    if (digest(context['actual_profile']) != context['actual_profile_sha256']
            or digest(meta['profile']) != context['actual_profile_sha256']
            or meta.get('source_path') != context['source_path']):
        raise ValueError('录音转换配置或来源不匹配')
    if (file_hash(folder / 'reference_processed.wav') != meta.get('calibration_reference_sha256')
            or file_hash(capture_path) != meta.get('calibration_capture_sha256')):
        raise ValueError('录音或参考音频已改变，不能用于本曲拟合')
    if (not meta.get('completed') or meta.get('paused') or meta.get('stopped') or meta.get('error')
            or recording.get('clipped_fraction', 1) > 0 or recording.get('portaudio_status')):
        raise ValueError('拟合需要无削波、无暂停或异常的完整同步录音')
    return copy.deepcopy(context), meta


def make_record(context, fitted_profile, recording_path, analysis=None):
    return dict(schema=SCHEMA, id=uuid.uuid4().hex, source_sha256=context['source_sha256'],
                base_profile_sha256=context['base_profile_sha256'], model_sha256=context['model_sha256'],
                profile=copy.deepcopy(fitted_profile), profile_sha256=digest(fitted_profile),
                recording_file=str(recording_path), recording_context=copy.deepcopy(context),
                analysis=copy.deepcopy(analysis))
