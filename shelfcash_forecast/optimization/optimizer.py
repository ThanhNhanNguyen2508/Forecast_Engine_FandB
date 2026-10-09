"""Bounded candidate generation; independent exact critic is the only acceptance gate."""
from __future__ import annotations
import time
import logging
from shelfcash_forecast.exceptions import OptimizationNotAvailableError
from shelfcash_forecast.optimization.contracts import OptimizationRequest, OptimizationResult, CandidateEvaluation, ProcurementDiagnostic
from shelfcash_forecast.optimization.deterministic import solve_deterministic_procurement
from shelfcash_forecast.optimization.stochastic import solve_stochastic_procurement
from shelfcash_forecast.optimization.resimulation import evaluate_candidate_plan
from shelfcash_forecast.optimization.strategies import default_strategy_profiles
from shelfcash_forecast.optimization.preflight import preflight
from shelfcash_forecast.optimization.planning_service import content_hash


def _profiles(request):
    defaults = {p.name:p for p in default_strategy_profiles()}
    defaults.update({p.name:p for p in request.strategy_profiles})
    return [defaults[name] for name in ("LEAN","BALANCED","PROTECTED")]


def _optimize_procurement(request, *, lead_time_model=None, shelf_life_model=None):
    started=time.monotonic(); profiles=_profiles(request)
    weighted=bool(request.demand_scenarios) and all(s.probability_weight is not None for s in request.demand_scenarios)
    stochastic=request.stochastic and weighted and len(request.demand_scenarios)>1
    fallback=None
    if request.stochastic and not stochastic:
        fallback="STOCHASTIC_NOT_AVAILABLE_WITH_UNWEIGHTED_OR_SINGLE_SCENARIO"
        if not request.allow_mode_fallback:
            raise OptimizationNotAvailableError("Stochastic mode requires at least two fully weighted scenarios.",code=fallback)
    actual_mode="stochastic" if stochastic else "deterministic"
    pf=preflight(request,profiles)
    if lead_time_model is not None or shelf_life_model is not None:
        pf["status"]="BLOCKED_INPUT_SEMANTICS"
        pf["diagnostics"].append(ProcurementDiagnostic(reason_code="EXTERNAL_SUPPLY_UNCERTAINTY_NOT_SUPPORTED",proof_status="BLOCKED"))
    base_provenance={"candidate_engine":"lot_fefo_milp_v2","requested_mode":"stochastic" if request.stochastic else "deterministic",
        "actual_mode":actual_mode,"validation_engine":"m4_lot_level_fefo_v1",
        "recommendation_rule":"BALANCED_then_PROTECTED_then_LEAN_if_valid", "preflight":{k:v for k,v in pf.items() if k!="diagnostics"},
        "proof_scope":pf["proof_scope"],
        "request_hash":content_hash(request),"scenario_provenance":request.scenario_provenance,
        "optimizer_called":False,"environment":request.environment,"business_pending":request.business_issues,
        "execution_authorized":False,"business_ready":False}
    if pf["status"] in {"BLOCKED_INPUT_SEMANTICS","PROVEN_INFEASIBLE"}:
        return OptimizationResult(request_id=request.request_id,evaluations={},status="NO_VALID_PROCUREMENT_PLAN",
            technical_outcome=pf["status"],technical_feasible=False if pf["status"]=="PROVEN_INFEASIBLE" else None,
            diagnostics=pf["diagnostics"],provenance=base_provenance)
    evaluations={};limited=False;candidate_modes={};full_joint_proofs=set()
    for profile in profiles:
        current=request;attempts=[];fingerprints=set();evaluation=None
        for attempt in range(request.limits.max_refinement_iterations+1):
            remaining=request.limits.total_seconds-(time.monotonic()-started)
            if remaining<=0:
                limited=True;break
            limits={**current.limits.model_dump(),"per_solve_seconds":min(current.limits.per_solve_seconds,remaining),"total_seconds":remaining}
            if current.candidate_generation == 'DECOMPOSED_FIXED_CERTIFICATION' and current.limits.joint_fallback:
                limits['total_seconds'] = remaining * (1-current.limits.fallback_reserve_fraction)
                limits['per_solve_seconds'] = min(limits['per_solve_seconds'], limits['total_seconds'])
            current=OptimizationRequest.model_validate({**current.model_dump(),"limits":limits})
            logging.getLogger(__name__).info('M5 %s %s attempt=%s worlds=%s solve',actual_mode,profile.name,attempt,len(current.demand_scenarios))
            if current.candidate_generation=='DECOMPOSED_FIXED_CERTIFICATION':
                from shelfcash_forecast.optimization.decomposition import solve_decomposed
                plan=solve_decomposed(current,profile,stochastic=stochastic)
            else:
                plan=(solve_stochastic_procurement if stochastic else solve_deterministic_procurement)(current,profile)
            base_provenance["optimizer_called"]=True
            evaluation=evaluate_candidate_plan(plan,current,profile)
            logging.getLogger(__name__).info('M5 %s %s attempt=%s solver=%s critic=%s violations=%s',actual_mode,profile.name,attempt,plan.solver_status,evaluation.critic.passed,evaluation.critic.hard_violations)
            fingerprint=content_hash([o.model_dump(mode="json") for o in plan.orders])
            record={"attempt":attempt,"request_hash":content_hash(current),"optimization_ids":[s.scenario_id for s in current.demand_scenarios],
                    "optimization_weights":{s.scenario_id:s.probability_weight for s in current.demand_scenarios},
                    "candidate_fingerprint":fingerprint,"plan":plan.model_dump(mode="json"),
                    "physics_simulation":evaluation.physics_simulation.model_dump(mode="json") if evaluation.physics_simulation else None,
                    "evaluation_simulation":evaluation.simulation.model_dump(mode="json") if evaluation.simulation else None,
                    "critic":evaluation.critic.model_dump(mode="json"),"termination":None}
            attempts.append(record)
            if evaluation.critic.passed:
                record["termination"]="ACCEPTED_BY_EXACT_CRITIC";break
            if plan.solver_status!="OPTIMAL":
                record["termination"]="SOLVER_"+plan.solver_status
                limited |= plan.solver_status=="LIMIT_REACHED";break
            if fingerprint in fingerprints:
                record["termination"]="CYCLE_DETECTED";limited=True;break
            fingerprints.add(fingerprint)
            if attempt==request.limits.max_refinement_iterations:
                record["termination"]="MAX_REFINEMENT_ITERATIONS";limited=True;break
            if not request.evaluation_scenarios or evaluation.simulation is None:
                record["termination"]="NO_SUPPORTED_REFINEMENT";break
            if "CANDIDATE_MODEL_MISMATCH" in evaluation.critic.hard_violations:
                record["termination"]="PHYSICS_MISMATCH_REQUIRES_MODEL_FIX";break
            existing={s.scenario_id for s in current.demand_scenarios}
            violating={r.scenario_id for r in evaluation.simulation.results if any(k.shortage_quantity>request.inventory_policy.accounting_tolerance for k in r.summary.by_key)}
            added=sorted(violating-existing)
            if not added:
                record["termination"]="NO_NEW_VIOLATING_WORLDS";break
            selected=[s for s in request.evaluation_scenarios if s.scenario_id in existing|set(added)]
            mass=sum(s.probability_weight for s in selected)
            if mass<=0:
                record["termination"]="ZERO_SELECTED_PROBABILITY_MASS";break
            worlds=[type(s).model_validate({**s.model_dump(),"probability_weight":s.probability_weight/mass}) for s in selected]
            current=OptimizationRequest.model_validate({**request.model_dump(),"demand_scenarios":worlds})
            record["added_scenarios"]=added;record["termination"]="ADD_EXISTING_VIOLATING_WORLDS"
        # A restricted candidate, mean world, or subset failure is never the
        # final feasibility search. Release EVERY purchase decision and evaluate
        # the unchanged full pool under the original policy.
        remaining=request.limits.total_seconds-(time.monotonic()-started)
        if (evaluation is None or not evaluation.critic.passed) and request.limits.joint_fallback and remaining > 0:
            from shelfcash_forecast.optimization.lot_milp import solve_lot_procurement
            pool=request.evaluation_scenarios or request.demand_scenarios
            joint=OptimizationRequest.model_validate({**request.model_dump(),
                'candidate_generation':'JOINT_MILP','demand_scenarios':[s.model_dump() for s in pool],
                'limits':{**request.limits.model_dump(),'total_seconds':remaining,
                    'per_solve_seconds':min(request.limits.per_solve_seconds,remaining)}})
            plan=solve_lot_procurement(joint,profile,stochastic=True,feasibility_first=True)
            evaluation=evaluate_candidate_plan(plan,joint,profile)
            attempts.append({'attempt':len(attempts),'phase':'FULL_POOL_UNFIXED_FEASIBILITY',
                'request_hash':content_hash(joint),'optimization_ids':[s.scenario_id for s in pool],
                'optimization_weights':{s.scenario_id:s.probability_weight for s in pool},
                'plan':evaluation.plan.model_dump(mode='json'),'critic':evaluation.critic.model_dump(mode='json'),
                'termination':'ACCEPTED_BY_EXACT_CRITIC' if evaluation.critic.passed else 'SOLVER_'+plan.solver_status,
                'fixed_commitments':False,'original_policy_preserved':True})
            limited |= plan.solver_status == 'LIMIT_REACHED'
            if plan.solver_status == 'INFEASIBLE': full_joint_proofs.add(profile.name)
        if evaluation is not None:
            candidate_actual_mode='stochastic' if evaluation.plan.provenance.get('search_phase')=='FEASIBILITY_FIRST' else actual_mode
            candidate_fallback='FULL_POOL_UNFIXED_FEASIBILITY' if candidate_actual_mode!=actual_mode else fallback
            evaluation=CandidateEvaluation.model_validate({**evaluation.model_dump(),"requested_mode":base_provenance["requested_mode"],
                "actual_mode":candidate_actual_mode,"fallback_reason":candidate_fallback,"attempts":attempts})
            evaluations[profile.name]=evaluation
            candidate_modes[profile.name]={"actual_mode":candidate_actual_mode,"solver_status":evaluation.plan.solver_status,
                                            "termination":attempts[-1]["termination"] if attempts else "TOTAL_TIME_LIMIT"}
    recommended=next((n for n in ("BALANCED","PROTECTED","LEAN") if n in evaluations and evaluations[n].critic.passed),None)
    proven = len(full_joint_proofs) == len(profiles)
    if recommended:base_provenance['actual_mode']=evaluations[recommended].actual_mode
    solver_error = any(e.plan.solver_status == 'SOLVER_ERROR' for e in evaluations.values())
    outcome="FEASIBLE" if recommended else "PROVEN_INFEASIBLE" if proven else "SOLVER_ERROR" if solver_error else "SEARCH_LIMIT_REACHED" if limited else "REJECTED_BY_EXACT_CRITIC"
    base_provenance.update(candidate_modes=candidate_modes,elapsed_seconds=time.monotonic()-started,
                           proof_scope="declared_lot_model_opportunities_profiles_and_worlds_including_deterministic_mean; not_unknown_real_supplier_semantics")
    return OptimizationResult(request_id=request.request_id,evaluations=evaluations,recommended_strategy=recommended,
        status="COMPLETED" if recommended else "NO_VALID_PROCUREMENT_PLAN",technical_outcome=outcome,
        technical_feasible=True if recommended else None if outcome in {'SEARCH_LIMIT_REACHED','SOLVER_ERROR'} else False,
        provenance=base_provenance,diagnostics=pf["diagnostics"],warnings=[fallback] if fallback else [])


def optimize_procurement(request, *, lead_time_model=None, shelf_life_model=None, structured_errors=False):
    """Public typed/dict boundary: invalid or unsupported input is a diagnostic result."""
    from pydantic import ValidationError
    from shelfcash_forecast.optimization.input_validation import validate_supported_request, error_result
    legacy_typed=isinstance(request,OptimizationRequest)
    try:
        request = OptimizationRequest.model_validate(request)
    except ValidationError as exc:
        return error_result(request, 'INVALID_INPUT', exc)
    try:
        issues = validate_supported_request(request)
        if issues:
            return OptimizationResult(request_id=request.request_id,evaluations={},status='NO_VALID_PROCUREMENT_PLAN',
                technical_outcome='BLOCKED_INPUT_SEMANTICS' if any(d.reason_code in {'SUPPLY_CHRONOLOGY_REQUIRED','UNIT_CONVERSION_REQUIRED'} for d in issues) else 'UNSUPPORTED_INPUT',diagnostics=issues)
        return _optimize_procurement(request,lead_time_model=lead_time_model,shelf_life_model=shelf_life_model)
    except OptimizationNotAvailableError as exc:
        if legacy_typed and not structured_errors:raise
        return error_result(request,'UNSUPPORTED_INPUT',exc)
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return error_result(request,'SOLVER_ERROR',exc)
