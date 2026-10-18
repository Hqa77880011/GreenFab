import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;k=c.get('outcomes',4);self.net=nn.Sequential(nn.Linear(c.get('input_shape',[8])[-1],32),nn.SiLU(),nn.Linear(32,3*k))
    def forward(self,b):
        mean,lo,hi=self.net(b['x'].float()).chunk(3,-1);return {'prediction':mean,'lower':mean-F.softplus(lo),'upper':mean+F.softplus(hi)}
    def objective(self,b,p):
        y=b['y'].float();loss=F.mse_loss(p['prediction'],y)
        for q,key in [(.05,'lower'),(.95,'upper')]:
            residual=y-p[key];loss=loss+torch.maximum(q*residual,(q-1)*residual).mean()
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,8)).astype('float32');return {'x':x,'y':np.stack([x[:,0],x[:,1]**2,x[:,2]+x[:,3],x[:,4]*x[:,5]],1).astype('float32'),'group':np.arange(n)}


def optimize(problem,variant='full'):
    from scipy.optimize import linprog
    outcomes=np.asarray(problem['objectives']);count,scenarios,k=outcomes.shape;prob=np.asarray(problem['probabilities']);distance=np.asarray(problem['distance']);robust=np.zeros((count,k))
    equality=np.zeros((scenarios,scenarios*scenarios))
    for source in range(scenarios):equality[source,source*scenarios:(source+1)*scenarios]=1
    for a in range(count):
        for j in range(k):
            if variant in {'baseline','no_robustness'}:robust[a,j]=prob@outcomes[a,:,j];continue
            result=linprog(-np.tile(outcomes[a,:,j],scenarios),A_ub=distance.reshape(1,-1),b_ub=[problem['radius']],A_eq=equality,b_eq=prob,bounds=(0,None),method='highs')
            if not result.success:raise ValueError('Wasserstein scenario transport is infeasible: '+result.message)
            robust[a,j]=-result.fun
    constraints=[]
    if variant!='no_preferences':
        for preferred,other in problem.get('preferences',[]):constraints.append(robust[preferred]-robust[other])
    admissible=problem.get('feasible',[True]*count);regret=[]
    for a in range(count):
        if not admissible[a]:regret.append(None);continue
        worst=0.
        for rival in range(count):
            if not admissible[rival]:continue
            result=linprog(-(robust[a]-robust[rival]),A_ub=np.array(constraints) if constraints else None,b_ub=np.zeros(len(constraints)) if constraints else None,A_eq=np.ones((1,k)),b_eq=[1],bounds=(0,1),method='highs')
            if not result.success:raise ValueError('Preference answers are mutually inconsistent')
            worst=max(worst,-result.fun)
        regret.append(float(worst))
    allowed=[i for i,r in enumerate(regret) if r is not None]
    if not allowed:raise ValueError('No physically feasible allocation candidate')
    selected=min(allowed,key=lambda i:regret[i]);return {'selected':selected,'robust_objectives':robust.tolist(),'minimax_regret':regret,'preference_constraints':[row.tolist() for row in constraints]}


def execute(args):
    import json
    from pathlib import Path
    result=optimize(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'allocation.json').write_text(json.dumps(result,indent=2))


def scenario_problem(prediction, data, expansion):
    """Build finite-scenario minimization objectives from calibrated upper bounds."""
    if not np.isfinite(expansion) or expansion<0:
        raise ValueError('Finite calibrated scenario bounds require enough independent calibration groups')
    objectives=np.asarray(prediction['upper'],dtype=float)+expansion
    count,scenarios,outcomes=objectives.shape
    probabilities=np.asarray(data['probabilities'],dtype=float)
    distance=np.asarray(data['distance'],dtype=float)
    feasible=np.asarray(data['feasible'])
    radius=float(np.asarray(data['radius']).item())
    if not np.isfinite(objectives).all() or not np.isfinite(probabilities).all() or probabilities.shape!=(scenarios,) or (probabilities<0).any() or not np.isclose(probabilities.sum(),1):
        raise ValueError('Finite objective cube and normalized scenario probabilities are required')
    if distance.shape!=(scenarios,scenarios) or not np.isfinite(distance).all() or (distance<0).any() or not np.allclose(distance,distance.T) or not np.allclose(np.diag(distance),0):
        raise ValueError('Scenario distance must be finite symmetric nonnegative with zero diagonal')
    if feasible.shape!=(count,) or not np.isin(feasible,[0,1]).all() or not np.isfinite(radius) or radius<0:
        raise ValueError('Supply a physical feasible mask and nonnegative Wasserstein radius')
    preferences=np.asarray(data.get('preferences',np.empty((0,2),dtype=int)))
    if preferences.shape!=(len(preferences),2) or not np.isin(preferences,np.arange(count)).all():
        raise ValueError('Preferences must be pairs of candidate indices')
    return {'objectives':objectives.tolist(),'probabilities':probabilities.tolist(),
            'distance':distance.tolist(),'radius':radius,'feasible':feasible.astype(bool).tolist(),
            'preferences':preferences.astype(int).tolist(),'objectives_source':'calibrated_surrogate_upper',
            'outcomes':outcomes}


def export(args):
    """Export model predictions on candidate-by-scenario features to allocation JSON."""
    import json
    from pathlib import Path
    state=torch.load(args.model,map_location=args.device,weights_only=True)
    if 'interval_expansion' not in state:
        raise ValueError('Calibrate the surrogate on independent fab-day groups before export')
    model=Model(state['config']).to(args.device);model.load_state_dict(state['state']);model.eval()
    with np.load(args.input,allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    x=np.asarray(data['x'])
    if x.ndim!=3 or not np.isfinite(x).all():
        raise ValueError('Scenario x must be finite candidates,scenarios,features')
    if 'group' in data:
        excluded=state.get('training_groups',[])+state.get('validation_groups',[])+state.get('calibration_groups',[])
        if np.intersect1d(data['group'],excluded).size:
            raise ValueError('Decision/evaluation scenario groups overlap surrogate fitting or calibration')
    with torch.no_grad():
        prediction=model({'x':torch.as_tensor(x.reshape(-1,x.shape[-1]),device=args.device)})
    upper=prediction['upper'].cpu().numpy().reshape(*x.shape[:2],-1)
    problem=scenario_problem({'upper':upper},data,state['interval_expansion'])
    candidate_ids=np.asarray(data.get('candidate_ids',np.arange(len(x))))
    scenario_ids=np.asarray(data.get('scenario_ids',np.arange(x.shape[1])))
    if candidate_ids.shape!=(len(x),) or scenario_ids.shape!=(x.shape[1],):
        raise ValueError('Candidate/scenario IDs must match their feature axes')
    problem['candidate_ids']=candidate_ids.tolist()
    problem['scenario_ids']=scenario_ids.tolist()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(problem,indent=2,allow_nan=False),encoding='utf-8')
