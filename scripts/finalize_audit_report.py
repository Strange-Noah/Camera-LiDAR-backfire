#!/usr/bin/env python3
"""Evidence-based research interpretation; preserves original automatic gate output."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
def read(name):return json.loads((out/name).read_text())
def rows(name):return list(csv.DictReader((out/name).open()))
def write(name,d):(out/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
import hashlib
assert hashlib.sha256(json.dumps(read('audit_config.json'),sort_keys=True).encode()).hexdigest()=='b7ceb31d72ebea77d328a0c4f2e1bb4796c353aa6ca3ca1fe66876041b4ec3f6', 'This reviewed narrative is specific to the audited design configuration'
assert read('source_and_scope_validation.json')['source_row_counts']=={'scene':850,'sample':34149,'instance':64386,'sample_annotation':1166187}
g=read('gating_report.json')
if not g['final_gating_eligible']:raise SystemExit('Research conclusion is reserved for the 850-scene trainval run')
auto=out/'gating_report.automatic.json'
if not auto.exists():write(auto.name,g)
else:g=read(auto.name)
construct={r['variable']:r for r in rows('gates/constructability.csv')}
joint=rows('joint/distance_x_lateral_velocity_coverage.csv')
car=[r for r in rows('mismatch_proxy/r_regime_by_class.csv') if r['category']=='vehicle.car' and r['regime']=='r>=1']
projection=read('geometry/projection_scale_review.json')
class_flags=[r for r in rows('identifiability/class_dominance.csv') if float(r['top_class_fraction'])>=.8]
limitations=[
    'Pooled mismatch coverage does not transfer to every class/dt. In vehicle.car, 50 ms has no r>=1 observations; see class-conditioned coverage.',
    'Physical size is semantically confounded: the middle-width bin is dominated by vehicle.car. Use within-class comparisons and observed overlapping support, not pooled size effects.',
    'CAM_FRONT projected scale is valid only on its geometric-FOV subset; near-plane rejection and partial image boundaries create selection. Large unclipped box envelopes are not visible object widths.',
    'Distance and LiDAR point support are dependent (descriptive pooled Spearman about -0.652); control observation quality when comparing geometric regimes.',
    'Relative lateral motion is strongly influenced by ego translation and is not equivalent to moving-object speed. Ego rotation needs a separate axis or a full projection-based model.',
    'Rare semantic classes do not share the broad pooled support. Emergency ambulance and some mobility classes cannot support the same broad claims under the configured design thresholds.'
]
supported=[
    'All structurally essential requested variables are constructible on the full official 850-scene trainval split; object/relative velocity remains missing on 4,405 annotations rather than imputed.',
    'Distance/lateral-motion comparisons have broad scene and trajectory support: all nine global tertile cells meet the declared project support rules.',
    'Within-class coarse comparisons can hold distance/width approximately fixed while lateral motion varies, and conversely; overlap is evidence for designing comparisons, not a causal estimate.',
    'A class-conditioned temporal perturbation study is feasible. For vehicle.car, 100–200 ms supports all four specified physical-proxy bands; 50 ms is a lower-mismatch control axis.',
    'Native sensor offsets are predominantly camera-dependent with additional scene variation and outliers; actual timestamps should define future perturbations.'
]
unsupported=[
    'An existing fusion failure/backfire boundary, accuracy degradation curve, or validated cross-scene normalization law.',
    'Uniform large-mismatch coverage at 50 ms for vehicle.car, or pooled coverage generalized to all semantic classes.',
    'Causal separation of size, motion, distance and sensing quality from a correlation matrix or coarse quantile matching alone.',
    'Treating 1,166,187 annotations as independent trials, or treating counts of trajectories/scenes as a formally estimated effective sample size.',
    'Interpreting camera FOV projection as unoccluded camera visibility, or physical width W as an orientation-independent projected width.',
    'Treating the 0.5 s first-order proxy or finite-difference velocity as exact instantaneous geometry.'
]
next_step=('Proceed with a restricted experimental design, not automatic full sensor acquisition: start with well-supported classes such as vehicle.car and adult pedestrians; prespecify distance × lateral-motion strata within class and width strata, balance visibility/LiDAR support, and separate high-yaw scenes. Use 50/100/200 ms initially, with 500 ms as a clearly approximate stress test. Reserve held-out scenes (prefer also log-level separation), cluster summaries by trajectory/scene, and check clipped versus unclipped camera scale before testing normalization. Download only the sensor subsets needed for that prespecified design, then obtain explicit authorization for GPU/BEVFusion deployment. Current evidence does not require switching to Waymo/AV2; supplement only if a targeted unsupported class/regime or external generalization claim is required.')
g['overall']='CONDITIONAL GO' if g['overall']!='NO-GO' else g['overall']
g['limitations']=list(dict.fromkeys(g['limitations']+limitations))
g['supported_claims']=supported;g['unsupported_claims']=unsupported;g['recommended_next_step']=next_step
review={'status':g['overall'],'scope':'850 scenes only','automatic_output_preserved':'gating_report.automatic.json',
    'critical_class_specific_proxy_evidence':car,'class_dominance_flags':class_flags,
    'projection_scale_evidence':projection,'decision_basis':'Broad constructability/joint/independent support, but class/dt restrictions, sensing-quality dependence and geometric approximations prevent unrestricted claims.',
    'requires_additional_trainval_sensor_blobs_for_this_audit':False,
    'is_research_judgment_not_universal_statistical_rule':True}
g['research_review']=review;write('gating_report.json',g);write('research_review.json',review)
text=(out/'README.md').read_text()
# Idempotent: retain the automatic report as a reproducible starting point.
base=out/'README.automatic.md'
if not base.exists():base.write_text(text)
else:text=base.read_text()
def add(section,addition):
    global text
    start=text.index('## '+str(section)+'. ')
    end=text.find('\n## ',start+1)
    if end<0:end=len(text)
    text=text[:end]+'\n\n'+addition.strip()+'\n'+text[end:]
def replace(section,body):
    global text
    start=text.index('## '+str(section)+'. ');heading_end=text.index('\n',start)
    end=text.find('\n## ',heading_end+1)
    if end<0:end=len(text)
    text=text[:heading_end+1]+'\n'+body.strip()+'\n'+text[end:]
add(2,'Official split membership independently checked: **700 train + 150 val**, 34,149 samples. Source JSON row counts exactly match 64,386 instances and 1,166,187 annotations. See `source_and_scope_validation.json`. Mini numerical checks and the full run are separate; mini contributes no suitability evidence.')
add(3,'Object and relative velocities are valid for **1,161,782 / 1,166,187 annotations (99.6223%)**, spanning 62,083 instances and all 850 scenes. The 4,405 missing values remain NaN. Other critical metadata variables have full constructability on their respective populations. CAM_FRONT projected width is available for **331,904 annotations (28.4606%)**, 32,252 trajectories and 850 scenes; this is FOV-conditioned availability, not a 71.5% metadata failure rate.')
add(4,'Full-data sanity diagnostics retain rather than delete extremes: maximum object speed 48.825 m/s, ego speed 18.392 m/s and |ego yaw rate| 0.6663 rad/s. Nominally stationary barrier/cone speed P95 is 0.2152 m/s, indicating annotation/estimation noise matters near zero. `motion/high_speed_diagnostics.csv` and examples identify records to inspect before interpreting fast-motion tails. Object-speed and lateral-relative-speed medians differ markedly (0.0576 versus 1.7483 m/s); ego motion is a major contributor.')
add(5,f'Among valid CAM_FRONT projections, **{projection["partial_boundary_count"]:,}** are partially outside the image and **{projection["projected_width_over_image_width_count"]:,}** have unclipped width >1,600 px. Do not use such widths as measured visible box sizes. Keep boundary-conditioned sensitivity analyses for a later image-scale normalization.')
add(7,'Using the separately declared |yaw rate| thresholds, high-yaw (>0.1 rad/s) coverage is **116,327 annotations / 16,462 trajectories / 393 scenes**. It is not confined to one or two scenes. `gates/support_threshold_sensitivity.csv` varies the scene/trajectory requirements, without presenting any cutoff as a statistical law.')
add(8,'Camera median offsets (camera minus LiDAR, ms): BACK −10.749; BACK_LEFT −0.876; BACK_RIGHT −20.302; FRONT −35.862; FRONT_LEFT −43.472; FRONT_RIGHT −27.895. After subtracting each camera median, P05–P95 residuals are −2.079 to +1.582 ms; the pooled offset range is not mainly random broad jitter. Rare extremes remain (−117.877 to +49.992 ms). This supports camera-dependent nominal offsets plus residual variation, not perfect synchronization or a demonstrated periodic pattern.')
add(9,f'Distance × lateral-motion cells contain **{min(int(r["N_traj"]) for r in joint):,}–{max(int(r["N_traj"]) for r in joint):,} trajectories** and **{min(int(r["N_scene"]) for r in joint)}–{max(int(r["N_scene"]) for r in joint)} scenes** each. The largest single-scene contribution among these cells is **{max(float(r["top1_scene_fraction"]) for r in joint):.2%}**. Pooled coverage is broad, but the class × motion × distance table identifies rare-class exceptions.')
add(10,'Important descriptive Spearman pairs: distance–lateral motion **−0.138**, distance–LiDAR count **−0.652**, width–distance **+0.209**, |yaw rate|–lateral motion **+0.356**, distance–projected width **−0.477**. None exceeds the configured 0.7 flag, but this is not evidence of no confounding. In particular, **87.16%** of the middle-width bin consists of cars. The 64/70 qualifying matched strata include nine car strata in each direction. Ambulance cells are all SPARSE; several mobility and other rare categories have limited support. Restrict claims to supported classes and matched regimes.')
proxy_lines=['For **vehicle.car**, large normalized mismatch (`r>=1`) has the following class-specific support:','','| dt | N_ann | N_traj | N_scene |','|---|---:|---:|---:|']
for r in car:proxy_lines.append(f'| {int(float(r["delta_t_s"])*1000)} ms | {int(r["N_ann"]):,} | {int(r["N_traj"]):,} | {int(r["N_scene"]):,} |')
proxy_lines+=['','At 50 ms the pooled r>=1 band exists (6,865 / 2,382 / 359), but **none of it is vehicle.car**. This is precisely why pooled mismatch coverage cannot justify every class-specific experiment. See `mismatch_proxy/r_regime_by_class.csv`.']
add(11,'\n'.join(proxy_lines))
add(12,'Research review keeps the automatic recommendation but adds a stricter class/dt interpretation. Gates 1–4 and the pooled mismatch gate have broad support. Gate 5 requires within-class/quality-controlled comparisons; Gate 6 must be read conditionally on semantic class and dt. `research_review.json` preserves explicit evidence and restrictions. The automatic report is retained separately for transparency.')
replace(13,'**CONDITIONAL GO — proceed with a restricted, prespecified temporal-misalignment experiment.** The data do not falsify the feasibility of the central geometric question, but do not justify unrestricted cross-class or cross-scene normalization claims. This is research judgment grounded in the diagnostics, not a suitability score.')
replace(14,'\n'.join('- '+s for s in supported))
replace(15,'\n'.join('- '+s for s in unsupported+limitations))
replace(16,next_step+'\n\nNo BEVFusion, GPU provisioning, corruption or inference has been executed. No trainval Camera/LiDAR blobs were required for this audit. All row-level Parquet tables and large trajectory tables remain at `/workspace/results/metadata_audit/`; the repository contains code, configuration and small derived summaries. See `artifact_inventory.json` for persistent artifacts.')
(out/'README.md').write_text(text)
