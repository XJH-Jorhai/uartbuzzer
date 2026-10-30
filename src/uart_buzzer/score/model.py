"""Hardware-neutral finite-note fitter (NumPy; no devices).

The conditional temporal DP is exact for its selected palette and cost. Palette
selection is a bounded greedy heuristic, not a globally optimal transcription.
Confidence/salience are algorithm scores, not calibrated probabilities.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

SCORE_SCHEMA = 'finite-discrete-score-v1'
PROFILE_SCHEMA = 'finite-note-device-profile-v1'
LATTICE_SCHEMA = 'discrete-note-candidates-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def midi_to_hz(midi):
    return 440. * 2. ** ((np.asarray(midi, float) - 69.) / 12.)


def hz_to_midi(hz):
    return 69. + 12. * np.log2(np.asarray(hz, float) / 440.)


@dataclass(frozen=True)
class FitSettings:
    palette_size: int = 12
    automatic_palette: bool = True
    palette_target_coverage: float = .97
    palette_coverage_radius_semitones: float = .5
    palette_histogram_floor: float = .05
    palette_swap_passes: int = 4
    time_quantum_s: float = .02
    max_pitch_deviation_semitones: float = 2.
    pitch_sigma_semitones: float = .65
    response_weight: float = .6
    note_start_cost: float = .32
    onset_reward: float = .3
    jump_cost: float = .025
    inactive_note_cost: float = 5.
    active_rest_cost: float = 4.
    allow_octave_adaptation: bool = False
    octave_shift_penalty: float = .8
    requested_voices: int = 1


def condition_on_melody(lattice, reference_events, prior_weight=.85):
    """Use the frontend's selected track, never ground truth, as a melody prior.

    The device fitter may quantize/remap this track within its explicit bound;
    it must not select an unrelated accompaniment just because it is louder.
    """
    t,pitches,sal,tracks,duration=validate_lattice(lattice)
    if not 0 <= prior_weight <= 1:
        raise ValueError('Invalid melody-prior weight')
    reference=np.full(len(t),np.nan);anchor=np.zeros_like(sal);onsets=np.zeros(len(t));normalized=[];last_end=0.
    for event in reference_events:
        start=float(event.get('onset_s',event.get('start_s',-1.)))
        length=float(event['duration_s']);midi=float(event.get('midi',event.get('midi_pitch',float('nan'))))
        if not np.isfinite([start,length,midi]).all() or start<last_end-1e-8 or start<0 or length<=0 or start+length>duration+1e-7 or event.get('voice',0)!=0:
            raise ValueError('Melody prior must be finite non-overlapping one-voice events')
        mask=(t>=start-1e-8)&(t<start+length-1e-8)
        reference[mask]=midi
        anchor[mask]=np.exp(-.5*((pitches-midi)/.25)**2)
        onsets[int(np.argmin(abs(t-start)))]=1.
        last_end=start+length
        normalized.append(dict(start_s=start,duration_s=length,midi=midi,voice=0))
    raw=sal/np.maximum(sal.max(axis=1,keepdims=True),1e-12)
    active=np.isfinite(reference)
    result=dict(lattice)
    result.update(salience=(prior_weight*anchor+(1-prior_weight)*raw)*active[:,None],
                  activity=tracks['activity']*active,onset_strength=onsets,reference_midi=reference)
    result['provenance']=dict(lattice.get('provenance',{}),melody_prior='explicit_frontend_selected_events_not_GT',
                              melody_prior_events_sha256=digest(normalized),melody_prior_weight=prior_weight)
    return result


def validate_profile(profile):
    p = json.loads(json.dumps(profile, allow_nan=False))
    if p.get('schema') != PROFILE_SCHEMA or not p.get('device_id'):
        raise ValueError('Invalid device profile schema/id')
    notes = p.get('notes', [])
    if not notes or len(notes)>512 or len({n['id'] for n in notes}) != len(notes):
        raise ValueError('Profile requires unique finite note ids')
    for note in notes:
        if not isinstance(note['id'], str) or not note['id']:
            raise ValueError('Note ids must be nonempty strings')
        if not 0 < float(note['frequency_hz']) < 1e6:
            raise ValueError('Invalid positive note frequency')
        if not 0 < float(note.get('gain', 1.)) <= 1.:
            raise ValueError('Relative note gains must be in (0,1]')
        if not 0 <= float(note.get('confidence', 0.)) <= 1.:
            raise ValueError('Invalid response confidence')
    p['notes'] = sorted(notes, key=lambda x: (x['frequency_hz'], x['id']))
    levels = np.asarray(p.get('velocity_levels', [.25, .5, .75, 1.]), float)
    if levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all() or np.any(levels <= 0) or np.any(levels > 1) or np.any(np.diff(levels) <= 0):
        raise ValueError('Velocity levels must be unique ascending values in (0,1]')
    for name in ('min_note_s', 'min_rest_s'):
        if not np.isfinite(p.get(name, .04)) or not 0 <= p.get(name, .04) <= 10.:
            raise ValueError('Negative/nonfinite dwell time')
    if not np.isfinite(p.get('max_pitch_step_semitones', 48.)) or p.get('max_pitch_step_semitones', 48.) <= 0:
        raise ValueError('Invalid positive finite pitch transition limit')
    if int(p.get('max_voices', 1)) < 1:
        raise ValueError('Invalid voice capacity')
    if p.get('response_kind') not in ('synthetic_relative', 'relative_acoustic', 'manual_relative',
                                     'unmeasured_flat', 'electrical_current_proxy'):
        raise ValueError('Response kind must explicitly identify its physical meaning')
    return p


def validate_lattice(lattice):
    if lattice.get('schema') != LATTICE_SCHEMA:
        raise ValueError('Candidate lattice schema mismatch')
    t = np.asarray(lattice['frame_times_s'], float)
    pitches = np.asarray(lattice['midi_pitches'], float)
    sal = np.asarray(lattice['salience'], float)
    hop = float(lattice['hop_s'])
    duration = float(lattice.get('duration_s', t[-1] + hop if len(t) else 0.))
    if t.ndim != 1 or not len(t) or pitches.ndim != 1 or not len(pitches) or sal.shape != (len(t), len(pitches)):
        raise ValueError('Invalid lattice axes/shape')
    if not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0) or hop <= 0 or not np.isfinite(hop) or not np.isfinite(duration) or duration <= 0 or duration + 1e-9 < t[-1]:
        raise ValueError('Invalid candidate time axis/duration')
    if not np.isfinite(pitches).all() or np.any(np.diff(pitches) <= 0) or not np.isfinite(sal).all() or np.any(sal < 0):
        raise ValueError('Invalid finite pitch/salience axis')
    tracks = {}
    for key in ('activity', 'onset_strength', 'loudness'):
        a = np.asarray(lattice[key], float)
        if a.shape != t.shape or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
            raise ValueError('Invalid [0,1] candidate '+key)
        tracks[key] = a
    return t, pitches, sal, tracks, duration


def aggregate_lattice(lattice, quantum):
    t, pitches, sal, tracks, duration = validate_lattice(lattice)
    n = max(1, int(np.ceil(duration / quantum)))
    index = np.minimum((t / quantum + 1e-8).astype(int), n-1)
    count = np.bincount(index, minlength=n)
    out = np.zeros((n, len(pitches)))
    for j in range(len(pitches)):
        out[:, j] = np.bincount(index, weights=sal[:, j], minlength=n) / np.maximum(count, 1)
    features = {}
    for key, values in tracks.items():
        if key == 'onset_strength':
            z = np.zeros(n); np.maximum.at(z, index, values)
        else:
            z = np.bincount(index, weights=values, minlength=n) / np.maximum(count, 1)
        features[key] = z
    if 'reference_midi' in lattice:
        ref=np.asarray(lattice['reference_midi'],float)
        if ref.shape!=t.shape or np.isinf(ref).any():raise ValueError('Invalid melody reference axis')
        z=np.full(n,np.nan)
        for i in range(n):
            values=ref[(index==i)&np.isfinite(ref)]
            if len(values):
                unique,counts=np.unique(values,return_counts=True)
                z[i]=unique[int(np.argmax(counts))]  # mode, stable lower-pitch tie; never invent a midpoint
        features['reference_midi']=z
    return pitches, out, features, duration


def _response_arrays(profile, enabled):
    measured = profile['response_kind'] in ('synthetic_relative', 'relative_acoustic', 'manual_relative')
    use = enabled and measured
    gain = np.array([n.get('gain', 1.) for n in profile['notes']], float) if use else np.ones(len(profile['notes']))
    confidence = np.array([n.get('confidence', 0.) for n in profile['notes']], float) if use else np.zeros(len(profile['notes']))
    gain=np.where(confidence>=.5,gain,1.)
    return gain, confidence, bool(use)


def select_palette(pitches, salience, activity, loudness, profile, settings, response_enabled=True,cancel_check=None):
    """Greedy weighted pitch coverage. Response cannot bypass a hard pitch limit."""
    frequencies = np.array([x['frequency_hz'] for x in profile['notes']])
    midi = hz_to_midi(frequencies)
    gains, confidence, use = _response_arrays(profile, response_enabled)
    # Palette evidence must survive the device's minimum dwell: a one-frame
    # false pitch should not take a dictionary slot away from a real melody.
    window = max(1, int(np.ceil(profile.get('min_note_s', .06)/settings.time_quantum_s)))
    window = min(window | 1, 7, max(1, (len(salience)//2)*2+1))
    padded = np.pad(salience, ((window//2,window//2),(0,0)), mode='edge')
    supported = np.median(np.lib.stride_tricks.sliding_window_view(padded,window,axis=0),axis=-1)
    hist = np.sum(supported * activity[:, None], axis=0)
    hist[hist < settings.palette_histogram_floor * hist.max(initial=0.)] = 0.
    if hist.sum() == 0:
        hist = np.ones(len(pitches))
    hist = hist / hist.sum()
    response_cost = settings.response_weight * confidence * (-.3*np.log(np.maximum(gains, .03))) if use else np.zeros(len(midi))
    shifts = (0, -12, 12) if settings.allow_octave_adaptation else (0,)
    options = []
    for shift in shifts:
        delta = abs(pitches[:, None] + shift - midi[None, :])
        costs = (delta / max(settings.pitch_sigma_semitones, .05))**2 / 2 + response_cost[None, :]
        costs[delta > settings.max_pitch_deviation_semitones+1e-9] = 12.
        options.append((float(np.sum(hist * costs.min(axis=1))) + settings.octave_shift_penalty*abs(shift)/12,
                        shift, costs))
    _, shift, costs = min(options, key=lambda x: (x[0], abs(x[1]), x[1]))
    exact = abs(pitches[:,None]+shift-midi[None,:]) <= settings.palette_coverage_radius_semitones+1e-9
    def summary(selection):
        loss = float(np.sum(hist*np.min(costs[:,selection],axis=1))) if selection else 12.
        coverage = float(np.sum(hist*np.any(exact[:,selection],axis=1))) if selection else 0.
        # Default prioritizes faithful coverage. Response cost is bounded to the
        # tie-break; after target coverage it cannot sacrifice that threshold.
        primary = -min(coverage,settings.palette_target_coverage) if settings.automatic_palette else 0.
        return (primary,loss),coverage
    selected = []
    for _ in range(min(settings.palette_size, len(midi))):
        if cancel_check and cancel_check():raise InterruptedError('Palette selection cancelled')
        candidates = [j for j in range(len(midi)) if j not in selected]
        winner = min(candidates,key=lambda j:(summary(selected+[j])[0],j))
        old_key,old_coverage=summary(selected);new_key,new_coverage=summary(selected+[winner])
        if selected and new_key >= old_key:
            break
        selected.append(winner)
        if settings.automatic_palette and new_coverage >= settings.palette_target_coverage-1e-10:
            break
    for _ in range(settings.palette_swap_passes):
        old_key,_=summary(selected);best_key=old_key;best=None
        for remove in selected:
            if cancel_check and cancel_check():raise InterruptedError('Palette refinement cancelled')
            for add in range(len(midi)):
                if add in selected:continue
                trial=sorted([j for j in selected if j!=remove]+[add])
                key,_=summary(trial)
                if key<best_key:best_key=key;best=trial
        if best is None:break
        selected=best
    selected.sort(key=lambda j: midi[j])
    _,coverage=summary(selected)
    return selected, shift, dict(method='dwell_supported_coverage_then_greedy_facility_local_swap_not_global_optimum',
                                weighted_coverage_cost=float(np.sum(hist*np.min(costs[:,selected],axis=1))),
                                weighted_exact_pitch_coverage=coverage,target_coverage=settings.palette_target_coverage,
                                target_reached=coverage>=settings.palette_target_coverage-1e-10,
                                supported_histogram_pitch_count=int(np.count_nonzero(hist)),
                                histogram_median_frames=window,
                                response_used=use, max_pitch_deviation_semitones=settings.max_pitch_deviation_semitones)


def fit_score(lattice, device_profile, settings=None, response_enabled=True, reference_events=None,
              cancel_check=None, progress=None):
    def cancelled():
        if cancel_check and cancel_check():raise InterruptedError('Finite-note fitting cancelled')
    cancelled()
    settings = FitSettings() if settings is None else settings
    if not isinstance(settings, FitSettings) or not 1 <= settings.palette_size <= 64 or not .005 <= settings.time_quantum_s <= .2:
        raise ValueError('Invalid finite palette/time quantum')
    if any(not np.isfinite(float(v)) for v in asdict(settings).values()):
        raise ValueError('Nonfinite fitter settings')
    if not isinstance(settings.palette_size, int) or not isinstance(settings.requested_voices, int) or not isinstance(settings.palette_swap_passes,int):
        raise ValueError('Palette and voices require integer counts')
    if not isinstance(settings.automatic_palette,bool) or not isinstance(settings.allow_octave_adaptation,bool):
        raise ValueError('Automatic palette/octave switches must be booleans')
    if not 0 < settings.palette_target_coverage <= 1 or not 0 <= settings.palette_histogram_floor < 1 or not 0 <= settings.palette_swap_passes <= 8:
        raise ValueError('Invalid bounded palette refinement settings')
    if settings.max_pitch_deviation_semitones <= 0 or settings.response_weight < 0:
        raise ValueError('Invalid fit weights/limit')
    p = validate_profile(device_profile)
    if reference_events is not None:
        lattice=condition_on_melody(lattice,reference_events)
    if settings.requested_voices != 1:
        raise NotImplementedError('Research v1 implements one voice; IR permits future voices')
    if settings.requested_voices > int(p.get('max_voices', 1)):
        raise ValueError('Requested voices exceed physical capacity')
    pitches, sal, tracks, duration = aggregate_lattice(lattice, settings.time_quantum_s)
    cancelled()
    if progress:progress(.12)
    planned_ages=max(1,int(np.ceil(max(p.get('min_note_s',.06),p.get('min_rest_s',.02))/settings.time_quantum_s)))
    if len(sal)*(min(settings.palette_size,len(p['notes']))+1)*planned_ages>12_000_000:
        raise ValueError('Research DP resource guard before palette/allocation')
    chosen, shift, palette_info = select_palette(pitches, sal, tracks['activity'], tracks['loudness'], p, settings, response_enabled,cancel_check)
    cancelled()
    if progress:progress(.25)
    notes = [p['notes'][i] for i in chosen]
    q = hz_to_midi([n['frequency_hz'] for n in notes]); count = len(notes); rest = count
    gains_all, confidence_all, use = _response_arrays(p, response_enabled)
    gains = gains_all[chosen]; confidence = confidence_all[chosen]
    levels = np.asarray(p.get('velocity_levels', [.25, .5, .75, 1.]), float)
    distance = abs(pitches[:, None] + shift - q[None, :])
    compatibility = np.exp(-.5*(distance/settings.pitch_sigma_semitones)**2)
    compatibility[distance > settings.max_pitch_deviation_semitones+1e-9] = 0.
    normalized = sal / np.maximum(sal.max(axis=1, keepdims=True), 1e-12)
    match = np.max(normalized[:, :, None] * compatibility[None, :, :], axis=1)
    emission = np.empty((len(sal), count+1))
    emission[:, :count] = -np.log(np.maximum(match, 1e-12)) + settings.inactive_note_cost*(1-tracks['activity'][:, None])
    if use:
        desired = tracks['loudness'][:, None, None]
        error = ((gains[None, :, None]*levels[None, None, :]-desired) / np.maximum(desired, .1))**2
        emission[:, :count] += settings.response_weight*confidence[None, :]*(np.min(error, axis=2)-.3*np.log(np.maximum(gains[None, :], .03)))
    emission[:, rest] = settings.active_rest_cost * tracks['activity']
    if 'reference_midi' in tracks:
        distance_to_melody=abs(tracks['reference_midi'][:,None]+shift-q[None,:])
        permitted=np.isfinite(distance_to_melody)&(distance_to_melody<=settings.max_pitch_deviation_semitones+1e-9)
        emission[:,:count][~permitted]=np.inf
    quantum = settings.time_quantum_s
    dwell = np.array([max(1, int(np.ceil(p.get('min_note_s', .06)/quantum)))]*count+
                     [max(1, int(np.ceil(p.get('min_rest_s', .02)/quantum)))])
    ages = int(dwell.max()); states = (count+1)*ages
    if len(sal)*states > 12_000_000:
        raise ValueError('Research DP resource guard: increase quantum or reduce duration/palette/dwell')
    previous = np.full((count+1, ages), np.inf)
    previous[rest, dwell[rest]-1] = 0.  # unrestricted silence before file start
    history = np.full((len(sal), states), -1, np.int32)
    resets = np.zeros((len(sal), states), bool)
    jump = np.zeros((count+1, count+1))
    jump[:count, :count] = abs(q[:, None]-q[None, :]) * settings.jump_cost
    maximum_jump = float(p.get('max_pitch_step_semitones', 48.))
    jump[:count, :count][abs(q[:, None]-q[None, :]) > maximum_jump+1e-9] = np.inf
    if p.get('require_rest_between_notes', False):
        jump[:count, :count] = np.inf
    for t in range(len(sal)):
        cancelled()
        if progress and t % max(1,len(sal)//100)==0:progress(.25+.7*t/len(sal))
        current = np.full_like(previous, np.inf)
        mature = previous[np.arange(count+1), dwell-1]
        for dest in range(count+1):
            # Continue an existing note/rest; cap age at its minimum dwell.
            for age in range(int(dwell[dest])):
                prev_age = max(age-1, 0)
                if age == dwell[dest]-1:
                    choices = [age] if age == 0 else [age-1, age]
                elif age == 0:
                    choices = []
                else:
                    choices = [prev_age]
                if choices:
                    winner = min(choices, key=lambda a: (previous[dest, a], a))
                    current[dest, age] = previous[dest, winner]+emission[t, dest]
                    history[t, dest*ages+age] = dest*ages+winner
            costs = mature + jump[:, dest]
            costs = costs.copy()
            if dest == rest:
                costs[rest] = np.inf
            else:
                costs += settings.note_start_cost*(1-tracks['onset_strength'][t])-settings.onset_reward*tracks['onset_strength'][t]
                if tracks['onset_strength'][t] < .65:
                    costs[dest] = np.inf
                if duration-t*quantum+1e-9 < p.get('min_note_s', .06):
                    costs[:] = np.inf
            source = int(np.argmin(costs)); value = costs[source]+emission[t, dest]
            if value < current[dest, 0]:
                current[dest, 0] = value
                history[t, dest*ages] = source*ages+dwell[source]-1
                resets[t, dest*ages] = True
        previous = current
    final_costs = previous[np.arange(count+1), dwell-1]
    last = int(np.argmin(final_costs)); objective = float(final_costs[last]); state = last*ages+dwell[last]-1
    path = np.empty(len(sal), int); reset_path = np.empty(len(sal), bool)
    for t in range(len(sal)-1, -1, -1):
        path[t] = state//ages; reset_path[t] = resets[t, state]
        state = int(history[t, state])
        if t and state < 0:
            raise RuntimeError('Invalid DP predecessor')
    boundaries = [0]
    for t in range(1, len(path)):
        if path[t] != path[t-1] or (path[t] != rest and reset_path[t]):
            boundaries.append(t)
    boundaries.append(len(path)); events = []; rests = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        start = a*quantum; end = min(b*quantum, duration)
        if end <= start:
            continue
        j = int(path[a])
        if j == rest:
            rests.append(dict(start_s=start, duration_s=end-start)); continue
        desired = float(np.median(tracks['loudness'][a:b]))
        level_index = int(np.argmin(abs(gains[j]*levels-desired)))
        confidence_score = float(np.mean(match[a:b, j]))
        events.append(dict(start_s=start, duration_s=end-start, note_id=notes[j]['id'],
                           frequency_hz=float(notes[j]['frequency_hz']), midi_pitch=float(q[j]),
                           velocity=float(levels[level_index]), velocity_index=level_index, voice=0,
                           confidence_score=confidence_score, confidence_kind='uncalibrated_compatibility'))
    score = dict(schema=SCORE_SCHEMA, version='finite-response-dp-product-v1', duration_s=duration,
                 time_quantum_s=quantum, velocity_levels=levels.tolist(), voices=1,
                 palette=[dict(id=n['id'], frequency_hz=float(n['frequency_hz']), midi_pitch=float(m)) for n,m in zip(notes,q)],
                 events=events, rests=rests, global_semitone_shift=shift,
                 device_profile_hash=digest(p), device_id=p['device_id'], backend=None,
                 fitting=dict(settings=asdict(settings), response_used=use,
                              response_kind=p['response_kind'], response_disabled_reason=None if use else 'no_supported_acoustic_response_or_explicit_generic_mode',
                              palette=palette_info, objective=objective,
                              objective_scope='conditional DP exact for selected palette; greedy palette; scores not probabilities'),
                 source_provenance=lattice.get('provenance', {}))
    validate_score(score, p)
    cancelled()
    if progress:progress(1.)
    return score


def validate_score(score, profile):
    p = validate_profile(profile)
    palette = {n['id']: n for n in score['palette']}
    allowed = {n['id']: n for n in p['notes']}
    if score.get('schema') != SCORE_SCHEMA or len(palette) != len(score['palette']):
        raise ValueError('Invalid Score IR')
    if len(palette) > score['fitting']['settings']['palette_size']:
        raise ValueError('Palette exceeds finite budget')
    previous_end = 0.
    previous_event=None
    for event in score['events']:
        if not np.isfinite([event['start_s'],event['duration_s'],event['velocity'],event['frequency_hz'],event['midi_pitch']]).all():
            raise ValueError('Nonfinite Score event')
        if event['note_id'] not in palette or event['note_id'] not in allowed or event['voice'] != 0:
            raise ValueError('Unsupported note/voice')
        if event['frequency_hz'] != allowed[event['note_id']]['frequency_hz']:
            raise ValueError('Event frequency does not match device note')
        if event['start_s']+1e-8 < previous_end or event['duration_s'] <= 0 or event['start_s']+event['duration_s'] > score['duration_s']+1e-8:
            raise ValueError('Overlapping or invalid note time')
        if event['duration_s']+1e-8 < p.get('min_note_s', .06):
            raise ValueError('Note shorter than device minimum dwell')
        if event['velocity'] not in score['velocity_levels']:
            raise ValueError('Velocity is not a finite supported level')
        if previous_event is not None:
            gap=event['start_s']-previous_end
            if gap<1e-8 and (p.get('require_rest_between_notes',False) or abs(event['midi_pitch']-previous_event['midi_pitch'])>p.get('max_pitch_step_semitones',48.)+1e-8):
                raise ValueError('Invalid direct device note transition')
            if 1e-8<gap<p.get('min_rest_s',.02)-1e-8:
                raise ValueError('Rest shorter than device minimum')
        previous_end = event['start_s']+event['duration_s']
        previous_event=event
    return True


def render_preview(score, sample_rate=16000):
    """Backend-neutral sine-note preview, not a physical device/SPL simulation."""
    x = np.zeros(int(round(score['duration_s']*sample_rate)), float)
    for event in score['events']:
        a = int(round(event['start_s']*sample_rate)); n = min(int(round(event['duration_s']*sample_rate)), len(x)-a)
        t = np.arange(n)/sample_rate
        envelope = np.minimum(t/.006, 1)*np.minimum(np.maximum(event['duration_s']-t, 0)/.012, 1)
        x[a:a+n] += .65*event['velocity']*envelope*np.sin(2*np.pi*event['frequency_hz']*t)
    return x


