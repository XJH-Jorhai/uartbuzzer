"""Optional native-speed pure numerical kernels; no devices or external solver."""
import numpy as np
import sys
import threading
_CACHE=not getattr(sys,'frozen',False)
try:
    from numba import njit
    COMPILED=True
except ImportError:
    COMPILED=False
    def njit(*args,**kwargs):
        if args and callable(args[0]):return args[0]
        return lambda f:f

@njit(cache=_CACHE,nogil=True)
def trajectory(data,F,g,H,q,initial):
    n=len(data);states=np.empty((n+1,6));values=np.empty((n,10));states[0]=initial
    for k in range(n):
        byte=int(data[k])
        for t in range(10):
            v=q[byte,t]
            for j in range(6):v+=H[t,j]*states[k,j]
            values[k,t]=v
        for i in range(6):
            v=g[byte,i]
            for j in range(6):v+=F[i,j]*states[k,j]
            states[k+1,i]=v
    return values,states

@njit(cache=_CACHE,nogil=True)
def reference(command,ad,bd,ah,bh,initial):
    n=len(command)//10;values=np.empty((n,10));states=np.empty((n+1,6));states[0]=initial;x=initial.copy()
    for k in range(n):
        for t in range(10):
            bit=command[k*10+t];v=bh[0]*bit
            for j in range(6):v+=ah[0,j]*x[j]
            values[k,t]=v;xx=np.empty(6)
            for i in range(6):
                v=bd[i]*bit
                for j in range(6):v+=ad[i,j]*x[j]
                xx[i]=v
            x=xx
        states[k+1]=x
    return values,states

@njit(cache=_CACHE,nogil=True)
def terminal_linear(errors,F,H,window):
    n=len(errors);back=np.zeros((n+1,6));r=np.zeros((n+1,6));fl=np.linalg.matrix_power(F,window)
    for k in range(n-1,-1,-1):
        for i in range(6):
            value=0.
            for t in range(10):value+=H[t,i]*errors[k,t]
            for j in range(6):value+=F[j,i]*back[k+1,j]
            back[k,i]=value
    for k in range(n+1):
        r[k]=back[k]
        if k+window<=n:
            for i in range(6):
                for j in range(6):r[k,i]-=fl[j,i]*back[k+window,j]
    return r

@njit(cache=_CACHE,nogil=True)
def beam_chunk(target,base,base_states,ref_states,F,g,H,q,gram,rlinear,initial,
               width,horizon,commit,candidates,terminal,terminal_weight,core_frames,
               action_cost,Hq,Q,target_projection,gPg):
    """All 256 action costs, deterministic Top-K, then optional action prescreen.
    Candidate/end states and paths are retained; actual plant state commits only the prefix.
    terminal: 0 none, 1 baseline-rollout, 2 relaxed reference state, 3 archived two-frame rollout.
    """
    result=np.empty(core_frames,np.uint8);x=initial.copy();expansions=0;n=len(base)
    for start in range(0,core_frames,commit):
        states=np.zeros((width,6));states[0]=x;costs=np.zeros(width);paths=np.zeros((width,horizon),np.uint8);count=1
        depth=min(horizon,n-start)
        for d in range(depth):
            idx=start+d;size=count*256
            ranks=np.full(size,np.inf);rawcost=np.empty(size);ends=np.empty((size,6))
            for parent in range(count):
                prop=np.zeros(6);common=costs[parent]
                for i in range(6):
                    common-=2*states[parent,i]*target_projection[idx,i]
                    for j in range(6):common+=states[parent,i]*Q[i,j]*states[parent,j]
                for i in range(6):
                    for j in range(6):prop[i]+=F[i,j]*states[parent,j]
                immediate=np.empty(256)
                for byte in range(256):
                    cost=common+action_cost[idx,byte]
                    for i in range(6):cost+=2*states[parent,i]*Hq[byte,i]
                    immediate[byte]=cost
                    pos=parent*256+byte;rawcost[pos]=cost
                allowed=np.zeros(256,np.uint8)
                if candidates==256:allowed[:]=1
                else:
                    # Cheap current-frame screen, with the retained baseline action mandatory.
                    screen=np.argpartition(immediate,candidates-1)[:candidates]
                    for choice in range(candidates):allowed[int(screen[choice])]=1
                    allowed[int(base[idx])]=1
                terminal_common=0.;terminal_linear_state=np.zeros(6)
                if terminal==1 or terminal==2:
                    anchor=base_states[idx+1] if terminal==1 else ref_states[idx+1]
                    delta=prop-anchor
                    for i in range(6):
                        for j in range(6):terminal_linear_state[i]+=gram[idx+1,i,j]*delta[j]
                        terminal_common+=delta[i]*terminal_linear_state[i]
                        if terminal==1:
                            terminal_common+=2*delta[i]*rlinear[idx+1,i]
                            terminal_linear_state[i]+=rlinear[idx+1,i]
                for byte in range(256):
                    if allowed[byte]==0:continue
                    pos=parent*256+byte;rank=rawcost[pos]
                    for i in range(6):ends[pos,i]=prop[i]+g[byte,i]
                    if terminal==1 or terminal==2:
                        value=terminal_common+gPg[idx+1,byte]
                        for i in range(6):
                            value+=2*g[byte,i]*terminal_linear_state[i]
                        rank+=terminal_weight*value
                    elif terminal==3:
                        forecast=ends[pos].copy()
                        for look in range(1,3):
                            k=idx+look
                            if k>=n:break
                            action=int(base[k]);future=0.
                            for t in range(10):
                                value=q[action,t]-target[k,t]
                                for j in range(6):value+=H[t,j]*forecast[j]
                                future+=value*value
                            rank+=.75*future;xx=np.empty(6)
                            for i in range(6):
                                value=g[action,i]
                                for j in range(6):value+=F[i,j]*forecast[j]
                                xx[i]=value
                            forecast=xx
                    ranks[pos]=rank;expansions+=1
            keep=min(width,size);newstates=np.empty((width,6));newcosts=np.empty(width);newpaths=np.zeros((width,horizon),np.uint8)
            for selected in range(keep):
                winner=int(np.argmin(ranks));ranks[winner]=np.inf;parent=winner//256;byte=winner%256
                newstates[selected]=ends[winner];newcosts[selected]=rawcost[winner]
                newpaths[selected]=paths[parent];newpaths[selected,d]=byte
            states=newstates;costs=newcosts;paths=newpaths;count=keep
        take=min(commit,core_frames-start)
        for j in range(take):
            byte=int(paths[0,j]);result[start+j]=byte;xx=np.empty(6)
            for i in range(6):
                v=g[byte,i]
                for k in range(6):v+=F[i,k]*x[k]
                xx[i]=v
            x=xx
    return result,x,expansions


# Research handoff: kernels_topw.py SHA256 18838ecf21de101b76f5c90fe0dd5cf6e0e5df974903c969139203a3568b8363.
# Project-original changes; private selector is imported from Numba (BSD).
beam_chunk_original=beam_chunk

SELECTOR_NUMBA_VERSION='0.62.1'
SELECTOR_SOURCE_SHA256='694eabf5ab6e67c4983fc5044494260f0f84714877af3c0090d44e8d0d133db5'
_selector=None
_numba_version=None
_scratch_errors=()
_backend_activity=threading.local()


def _scratch_support(version,helper,compiled):
    if not compiled:return False,'numba_unavailable'
    if version!=SELECTOR_NUMBA_VERSION:return False,'unsupported_numba_version'
    if not callable(helper):return False,'private_selector_unavailable'
    return True,None


if COMPILED:
    try:
        import numba as _numba
        _numba_version=_numba.__version__
        if _numba_version==SELECTOR_NUMBA_VERSION:
            from numba.np.arraymath import _arg_select_w_nan as _selector
            from numba.core.errors import TypingError,LoweringError,UnsupportedError,CompilerError,InternalError
            _scratch_errors=(TypingError,LoweringError,UnsupportedError,CompilerError,InternalError)
    except (ImportError,AttributeError):
        _selector=None
_arg_select_w_nan=_selector
_scratch_enabled,_scratch_reason=_scratch_support(_numba_version,_selector,COMPILED)


def backend_info():
    return dict(name='topw_selector_v1' if _scratch_enabled else 'original_compiled' if COMPILED else 'original_python',
                fallback_reason=_scratch_reason,numba_version=_numba_version,
                expected_selector_source_sha256=SELECTOR_SOURCE_SHA256,
                scratch_calls=getattr(_backend_activity,'scratch_calls',0),
                original_calls=getattr(_backend_activity,'original_calls',0))

@njit(cache=_CACHE,nogil=True)
def beam_chunk_topw(target,base,base_states,ref_states,F,g,H,q,gram,rlinear,initial,
               width,horizon,commit,candidates,terminal,terminal_weight,core_frames,
               action_cost,Hq,Q,target_projection,gPg):
    """All 256 action costs, deterministic Top-K, then optional action prescreen.
    Candidate/end states and paths are retained; actual plant state commits only the prefix.
    terminal: 0 none, 1 baseline-rollout, 2 relaxed reference state, 3 archived two-frame rollout.
    """
    result=np.empty(core_frames,np.uint8);x=initial.copy();expansions=0;n=len(base)
    states=np.empty((width,6));costs=np.empty(width);paths=np.empty((width,horizon),np.uint8)
    newstates=np.empty((width,6));newcosts=np.empty(width);newpaths=np.empty((width,horizon),np.uint8)
    ranks=np.empty(width*256);rawcost=np.empty(width*256);ends=np.empty((width*256,6))
    top_ranks=np.empty(width);top_indices=np.empty(width,np.intp)
    screen_indices=np.empty(256,np.intp)
    prop=np.empty(6);immediate=np.empty(256);allowed=np.empty(256,np.uint8)
    props=np.empty((width,6));HqT=np.empty((6,256));g2T=np.empty((6,256));terminal_values=np.empty(256)
    for i in range(6):
        for byte in range(256):
            HqT[i,byte]=Hq[byte,i]
            g2T[i,byte]=2*g[byte,i]
    terminal_linear_state=np.empty(6);delta=np.empty(6);next_x=np.empty(6)
    for start in range(0,core_frames,commit):
        states[:]=0.;states[0]=x;costs[:]=0.;paths[:]=0;count=1
        depth=min(horizon,n-start)
        for d in range(depth):
            idx=start+d;size=count*256
            ranks[:size]=np.inf;top_ranks[:]=np.inf;top_indices[:]=0;finite_ranks=0;all_finite=True
            for parent in range(count):
                prop[:]=0.;common=costs[parent]
                for i in range(6):
                    common-=2*states[parent,i]*target_projection[idx,i]
                    for j in range(6):common+=states[parent,i]*Q[i,j]*states[parent,j]
                for i in range(6):
                    for j in range(6):prop[i]+=F[i,j]*states[parent,j]
                # Reuse the same screen input; argpartition itself is unchanged.
                props[parent]=prop
                for byte in range(256):immediate[byte]=common+action_cost[idx,byte]
                for i in range(6):
                    state2=2*states[parent,i]
                    for byte in range(256):immediate[byte]+=state2*HqT[i,byte]
                for byte in range(256):rawcost[parent*256+byte]=immediate[byte]
                allowed[:]=0
                if candidates==256:allowed[:]=1
                else:
                    # Cheap current-frame screen, with the retained baseline action mandatory.
                    for byte in range(256):screen_indices[byte]=byte
                    _arg_select_w_nan(immediate,candidates-1,0,255,screen_indices)
                    for choice in range(candidates):allowed[int(screen_indices[choice])]=1
                    allowed[int(base[idx])]=1
                terminal_common=0.;terminal_linear_state[:]=0.
                if terminal==1 or terminal==2:
                    anchor=base_states[idx+1] if terminal==1 else ref_states[idx+1]
                    for i in range(6):delta[i]=prop[i]-anchor[i]
                    for i in range(6):
                        for j in range(6):terminal_linear_state[i]+=gram[idx+1,i,j]*delta[j]
                        terminal_common+=delta[i]*terminal_linear_state[i]
                        if terminal==1:
                            terminal_common+=2*delta[i]*rlinear[idx+1,i]
                            terminal_linear_state[i]+=rlinear[idx+1,i]
                if terminal==1 or terminal==2:
                    for byte in range(256):terminal_values[byte]=terminal_common+gPg[idx+1,byte]
                    for i in range(6):
                        for byte in range(256):terminal_values[byte]+=g2T[i,byte]*terminal_linear_state[i]
                for byte in range(256):
                    if allowed[byte]==0:continue
                    pos=parent*256+byte;rank=rawcost[pos]
                    if terminal==3:
                        for i in range(6):ends[pos,i]=prop[i]+g[byte,i]
                    if terminal==1 or terminal==2:
                        rank+=terminal_weight*terminal_values[byte]
                    elif terminal==3:
                        forecast=ends[pos].copy()
                        for look in range(1,3):
                            k=idx+look
                            if k>=n:break
                            action=int(base[k]);future=0.
                            for t in range(10):
                                value=q[action,t]-target[k,t]
                                for j in range(6):value+=H[t,j]*forecast[j]
                                future+=value*value
                            rank+=.75*future;xx=np.empty(6)
                            for i in range(6):
                                value=g[action,i]
                                for j in range(6):value+=F[i,j]*forecast[j]
                                xx[i]=value
                            forecast=xx
                    ranks[pos]=rank;expansions+=1
                    if not np.isfinite(rank):all_finite=False
                    else:
                        finite_ranks+=1
                        if rank<top_ranks[width-1]:
                            insert=width-1
                            while insert>0 and rank<top_ranks[insert-1]:
                                top_ranks[insert]=top_ranks[insert-1];top_indices[insert]=top_indices[insert-1];insert-=1
                            top_ranks[insert]=rank;top_indices[insert]=pos
            keep=min(width,size);newpaths[:]=0
            for selected in range(keep):
                if all_finite and finite_ranks>=keep:winner=int(top_indices[selected])
                else:winner=int(np.argmin(ranks[:size]))
                ranks[winner]=np.inf;parent=winner//256;byte=winner%256
                for i in range(6):newstates[selected,i]=props[parent,i]+g[byte,i]
                newcosts[selected]=rawcost[winner]
                newpaths[selected]=paths[parent];newpaths[selected,d]=byte
            states,newstates=newstates,states;costs,newcosts=newcosts,costs;paths,newpaths=newpaths,paths;count=keep
        take=min(commit,core_frames-start)
        for j in range(take):
            byte=int(paths[0,j]);result[start+j]=byte;xx=next_x
            for i in range(6):
                v=g[byte,i]
                for k in range(6):v+=F[i,k]*x[k]
                xx[i]=v
            x,next_x=xx,x
    return result,x,expansions

def beam_chunk(*args):
    """Exact original kernel remains available if the private backend cannot compile."""
    global _scratch_enabled,_scratch_reason
    # Nonfinite / extreme low-level diagnostics have no reproducible legacy
    # all-inf end-state semantics. Keep those calls on the original path.
    diagnostic=(len(args)==23 and
                (not np.all(np.isfinite(args[0])) or not np.all(np.isfinite(args[10]))
                 or np.max(np.abs(args[0]),initial=0.)>1e100
                 or np.max(np.abs(args[10]),initial=0.)>1e100))
    if _scratch_enabled and not diagnostic:
        try:
            result=beam_chunk_topw(*args)
        except _scratch_errors as error:
            _scratch_enabled=False
            _scratch_reason='scratch_compilation_failed:'+type(error).__name__
        else:
            _backend_activity.scratch_calls=getattr(_backend_activity,'scratch_calls',0)+1
            return result
    _backend_activity.original_calls=getattr(_backend_activity,'original_calls',0)+1
    return beam_chunk_original(*args)
