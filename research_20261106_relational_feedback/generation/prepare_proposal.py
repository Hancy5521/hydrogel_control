from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path(r'H:\polymer_completion_public\research\relational_feedback_20261106T061957384774Z');p=root/'generation';source=p/'joint_sampler.py'
fixture=root/'full_nig_bma'/'execution_once'/'RESULTS.json'
proposal={'status':'SOURCE_ONLY_BOUNDED_SAMPLING_PROPOSAL_NOT_EXECUTION_APPROVAL','actual_utc':datetime.now(timezone.utc).isoformat(),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'input_fixture':{'file':str(fixture),'sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'ensemble_key':'initial_ensemble','contexts':'fixed_fixture.query_context followed by fixed_fixture.target_context','response':'fixed_fixture.response_transform','rows_are_distinct_physical_units':True},
 'scope':'One previously frozen artificial 3NIG fixture, two jointly generated response rows; no real material labels, fitting or model updates.',
 'seed':20261106,'n_draws':131072,'n_draws_power_of_two':17,'return_auxiliary':True,
 'required_law':'Each draw samples exactly one H, one sigma2~IG(alpha_H,b_H), and one beta~Normal(m_H,sigma2 V_H) shared across both rows; independent residuals for the two distinct rows.',
 'known_reference_formulas':{'component_mean':'mu_h=Phi_h m_h','component_covariance':'Sigma_h=b_h/(alpha_h-1)*(I_N+Phi_h V_h Phi_h^T)','mixture_mean':'mu=sum_h w_h mu_h','mixture_covariance':'sum_h w_h (Sigma_h+(mu_h-mu)(mu_h-mu)^T)','marginal_CDF':'sum_h w_h T_(2alpha_h)((z-mu_hi)/sqrt((b_h/alpha_h)*(1+phi_hi V_h phi_hi)))','native_response':'native=s0*inverse_YJ_half(center+sd*z)'},
 'fixed_checks':{'modeled_CDF_points_per_row':[-2.,0.,2.],'native_CDF_thresholds':'native inverse map of the same six modeled thresholds; independently use forward YJ for verification','mixture_model_H_frequency_max_abs_probability_error':.005,'mixture_marginal_CDF_max_abs_probability_error':.005,'latent_mean_max_abs_error':.03,'latent_covariance_max_abs_error':.08,'auxiliary_standardized_row_residual_mean_max_abs':.02,'auxiliary_standardized_row_residual_covariance_max_abs_from_identity':.02,
 'input_state_context_and_response_unchanged':True,'one_H_index_per_draw_not_per_row':True,'one_sigma2_and_coefficient_vector_per_draw':True,'metadata_counters_exact':{'H_draws':131072,'noise_variance_draws':131072,'coefficient_vector_draws':131072,'row_residual_draws':262144,'model_updates':0,'data_source_reads':0}},
 'MC_interpretation':'Prechosen finite-draw numerical checks only. Report actual errors and MC uncertainty. Passing does not certify calibration, material effects or exact agreement of sample moments. Do not tune seed, n_draws or tolerances after outputs.',
 'why_latent_moments_only':'The existing nu6 law has finite latent fourth moments. Inverse-YJ native high-order empirical moments can be unstable; native marginal CDF remains the direct output check.',
 'independent_audit':'Independently rebuild Phi, mixture t CDF and analytic latent mean/covariance from pinned fixture; do not import sampler as a numerical oracle. Inspect perdraw shared H/sigma/coefficients and reconstruct each row residual from saved auxiliary draws.',
 'sample_count_is_not_materialN':True,'samples_are_statistical_generation_not_measurements':True,
 'no_new_material_roles_or_labels':True,'do_not_delay_primary4032law_stateaudit_or_96target_score':True,
 'requires_before_execution':'root actual UTC/source and input hashes, independent static review; no numerical execution by this preparation.'}
with (p/'SAMPLING_GATE_PROPOSAL.json').open('x',encoding='utf-8') as f:json.dump(proposal,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'source_sha256':proposal['source_sha256'],'fixture_sha256':proposal['input_fixture']['sha256'],'proposal_sha256':hashlib.sha256((p/'SAMPLING_GATE_PROPOSAL.json').read_bytes()).hexdigest()},indent=2))
