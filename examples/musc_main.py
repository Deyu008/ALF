import argparse
import os
import sys
import datetime
import subprocess
sys.path.append(os.getcwd())
from models.musc import MuSc
from utils.load_config import load_yaml

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='MuSc')
    parser.add_argument('--config', type=str, default='./configs/musc.yaml', help='config file path')
    parser.add_argument('--data_path', type=str, default=None, help='dataset path')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
    parser.add_argument('--class_name', type=str, default=None, help='category')
    parser.add_argument('--device', type=int, default=None, help='gpu id')
    parser.add_argument('--output_dir', type=str, default=None, help='save results path')
    parser.add_argument('--vis', type=str, default=None, help='visualization')
    parser.add_argument('--vis_type', type=str, default=None, help='normalization type in visualization')
    parser.add_argument('--save_excel', type=str, default=None, help='save excel')
    parser.add_argument('--r_list', type=int, nargs="+", default=None, help='aggregation degrees of LNAMD')
    parser.add_argument('--feature_layers', type=int, nargs="+", default=None, help='feature layers')
    parser.add_argument('--backbone_name', type=str, default=None, help='backbone')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained datasets')
    parser.add_argument('--img_resize', type=int, default=None, help='image size')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--divide_num', type=int, default=None, help='the number of divided subsets')
    parser.add_argument('--use_hier_fusion', type=str, default=None, help='enable hierarchical fusion')
    parser.add_argument('--hier_temperature', type=float, default=None, help='hier fusion temperature')
    parser.add_argument('--hier_top_p', type=float, default=None, help='hier fusion top-p sparsification')
    parser.add_argument('--hier_lambda_patch', type=float, default=None, help='hier fusion patch gate strength')
    parser.add_argument('--hier_patch_mode', type=str, default=None, help='hier fusion patch gate mode')
    parser.add_argument('--hier_gamma_img', type=float, default=None, help='hier fusion image score blend')
    parser.add_argument('--hier_cls_topk_ratio', type=float, default=None, help='hier fusion cls top-k ratio')
    parser.add_argument('--hier_adaptive_enabled', type=str, default=None, help='enable hier fusion adaptive tuning')
    parser.add_argument('--hier_adaptive_neff_floor', type=float, default=None, help='adaptive neff floor')
    parser.add_argument('--hier_adaptive_neff_ceil', type=float, default=None, help='adaptive neff ceil')
    parser.add_argument('--hier_adaptive_lambda_min', type=float, default=None, help='adaptive lambda min scale')
    parser.add_argument('--hier_adaptive_lambda_max', type=float, default=None, help='adaptive lambda max scale')
    parser.add_argument('--hier_adaptive_log_stats', type=str, default=None, help='adaptive log stats switch')
    parser.add_argument('--hier_adaptive_top_p_enabled', type=str, default=None, help='adaptive top-p switch')
    parser.add_argument('--hier_adaptive_top_p_min', type=float, default=None, help='adaptive top-p minimum')
    parser.add_argument('--hier_adaptive_top_p_max', type=float, default=None, help='adaptive top-p maximum')
    parser.add_argument('--hier_adaptive_temp_enabled', type=str, default=None, help='adaptive temperature switch')
    parser.add_argument('--hier_adaptive_temp_min', type=float, default=None, help='adaptive temperature minimum')
    parser.add_argument('--hier_adaptive_temp_max', type=float, default=None, help='adaptive temperature maximum')
    parser.add_argument('--dump_rescore_inputs', type=str, default=None, help='dump per-category inputs for image rescoring sweep')
    parser.add_argument('--dump_rescore_dir', type=str, default=None, help='optional directory name under output_dir to store dumps')
    parser.add_argument('--dump_rscin_hist', type=str, default=None, help='dump image-level scores before/after RsCIN (for histogram)')
    parser.add_argument('--dump_rscin_hist_dir', type=str, default=None, help='optional directory name under output_dir to store RsCIN dumps')
    parser.add_argument('--inductive_bank_max_images', type=int, default=None, help='limit number of normal-bank images (-1 means all)')
    parser.add_argument('--inductive_bank_sample_mode', type=str, default=None, help='normal-bank sampling mode: sequential|random')
    parser.add_argument('--inductive_bank_sample_seed', type=int, default=None, help='random sampling seed for normal-bank subset')

    # Train-free transductive refinement (worktree-compatible flags)
    parser.add_argument('--transductive_enabled', type=str, default=None, help='enable transductive refinement')
    parser.add_argument('--transductive_num_passes', type=int, default=None, help='number of transductive passes (e.g., 2)')
    parser.add_argument('--transductive_bank_ratio_mode', type=str, default=None, help='bank ratio mode: fixed | adaptive_knee')
    parser.add_argument('--transductive_bank_ratio', type=float, default=None, help='fallback bank ratio (used in fixed mode or knee fallback)')
    parser.add_argument('--transductive_bank_ratio_min', type=float, default=None, help='min clamp for bank ratio')
    parser.add_argument('--transductive_bank_ratio_max', type=float, default=None, help='max clamp for bank ratio')
    parser.add_argument('--transductive_bank_filter', type=str, default=None, help='enable bank filter (knn_mad)')
    parser.add_argument('--transductive_retrieval_topk', type=int, default=None, help='top-k retrieval within bank')
    parser.add_argument('--transductive_score_pool', type=str, default=None, help='pooling for Pass-0 image score: max|q99|topk_mean')
    parser.add_argument('--transductive_msm_topmin_min', type=float, default=None, help='MSM topmin_min for MSM_with_refs')
    parser.add_argument('--transductive_msm_topmin_max', type=float, default=None, help='MSM topmin_max for MSM_with_refs')
    parser.add_argument('--transductive_safe_fusion', type=str, default=None, help='enable safe fusion (Pass0/Pass1 blend)')
    parser.add_argument('--transductive_safe_fusion_weight', type=float, default=None, help='fixed fusion weight when adaptive_gate is off')
    parser.add_argument('--transductive_adaptive_gate', type=str, default=None, help='enable adaptive gate (confidence-based fusion)')
    parser.add_argument('--transductive_gate_threshold', type=float, default=None, help='gate threshold for fallback to Pass-0')
    parser.add_argument('--transductive_apply_to_pixel', type=str, default=None, help='apply Final maps to pixel-level (True)')
    parser.add_argument('--transductive_knee_strength_min', type=float, default=None, help='min knee strength to accept knee')
    parser.add_argument('--transductive_knee_tail_frac', type=float, default=None, help='tail frac to reject knee (fallback)')
    args = parser.parse_args()
    return args

def load_args(cfg, args):
    # If input new arguments through the script (musc.sh), the default configuration in the config file (musc.yaml) will be overwritten.
    if args.data_path is not None:
        cfg['datasets']['data_path'] = args.data_path
    assert os.path.exists(cfg['datasets']['data_path']), f"The dataset path {cfg['datasets']['data_path']} does not exist."
    if args.dataset_name is not None:
        cfg['datasets']['dataset_name'] = args.dataset_name
    if args.class_name is not None:
        cfg['datasets']['class_name'] = args.class_name
    if args.device is not None:
        cfg['device'] = args.device
    if isinstance(cfg['device'], int):
        cfg['device'] = str(cfg['device'])
    if args.output_dir is not None:
        cfg['testing']['output_dir'] = args.output_dir
    os.makedirs(cfg['testing']['output_dir'], exist_ok=True)
    if args.vis is not None:
        if args.vis.lower() == 'true':
            cfg['testing']['vis'] = True
        else:
            cfg['testing']['vis'] = False
    if args.vis_type is not None:
        cfg['testing']['vis_type'] = args.vis_type
    if args.save_excel is not None:
        if args.save_excel.lower() == 'true':
            cfg['testing']['save_excel'] = True
        else:
            cfg['testing']['save_excel'] = False
    if args.r_list is not None:
        cfg['models']['r_list'] = args.r_list
    if isinstance(cfg['models']['r_list'], int):
        cfg['models']['r_list'] = [cfg['models']['r_list']]
    if args.feature_layers is not None:
        cfg['models']['feature_layers'] = args.feature_layers
    if isinstance(cfg['models']['feature_layers'], int):
        cfg['models']['feature_layers'] = [cfg['models']['feature_layers']]
    if args.backbone_name is not None:
        cfg['models']['backbone_name'] = args.backbone_name
    if args.pretrained is not None:
        cfg['models']['pretrained'] = args.pretrained
    if args.img_resize is not None:
        cfg['datasets']['img_resize'] = args.img_resize
    if args.batch_size is not None:
        cfg['models']['batch_size'] = args.batch_size
    if args.divide_num is not None:
        cfg['datasets']['divide_num'] = args.divide_num
    if args.use_hier_fusion is not None:
        cfg['models']['use_hier_fusion'] = args.use_hier_fusion.lower() == 'true'
    hier_cfg = cfg['models'].setdefault('hier_fusion', {})
    if args.hier_temperature is not None:
        hier_cfg['temperature'] = args.hier_temperature
    if args.hier_top_p is not None:
        hier_cfg['top_p'] = args.hier_top_p
    if args.hier_lambda_patch is not None:
        hier_cfg['lambda_patch'] = args.hier_lambda_patch
    if args.hier_patch_mode is not None:
        hier_cfg['patch_mode'] = args.hier_patch_mode
    if args.hier_gamma_img is not None:
        hier_cfg['gamma_img'] = args.hier_gamma_img
    if args.hier_cls_topk_ratio is not None:
        hier_cfg['cls_topk_ratio'] = args.hier_cls_topk_ratio
    adaptive_cfg = hier_cfg.setdefault('adaptive', {})
    if args.hier_adaptive_enabled is not None:
        adaptive_cfg['enabled'] = args.hier_adaptive_enabled.lower() == 'true'
    if args.hier_adaptive_neff_floor is not None:
        adaptive_cfg['neff_floor'] = args.hier_adaptive_neff_floor
    if args.hier_adaptive_neff_ceil is not None:
        adaptive_cfg['neff_ceil'] = args.hier_adaptive_neff_ceil
    if args.hier_adaptive_lambda_min is not None:
        adaptive_cfg['lambda_min_scale'] = args.hier_adaptive_lambda_min
    if args.hier_adaptive_lambda_max is not None:
        adaptive_cfg['lambda_max_scale'] = args.hier_adaptive_lambda_max
    if args.hier_adaptive_log_stats is not None:
        adaptive_cfg['log_stats'] = args.hier_adaptive_log_stats.lower() == 'true'
    if args.hier_adaptive_top_p_enabled is not None:
        adaptive_cfg['top_p_enabled'] = args.hier_adaptive_top_p_enabled.lower() == 'true'
    if args.hier_adaptive_top_p_min is not None:
        adaptive_cfg['top_p_min'] = args.hier_adaptive_top_p_min
    if args.hier_adaptive_top_p_max is not None:
        adaptive_cfg['top_p_max'] = args.hier_adaptive_top_p_max
    if args.hier_adaptive_temp_enabled is not None:
        adaptive_cfg['temperature_enabled'] = args.hier_adaptive_temp_enabled.lower() == 'true'
    if args.hier_adaptive_temp_min is not None:
        adaptive_cfg['temperature_min'] = args.hier_adaptive_temp_min
    if args.hier_adaptive_temp_max is not None:
        adaptive_cfg['temperature_max'] = args.hier_adaptive_temp_max

    if args.dump_rescore_inputs is not None:
        cfg.setdefault('testing', {})
        cfg['testing']['dump_rescore_inputs'] = args.dump_rescore_inputs.lower() == 'true'
    if args.dump_rescore_dir is not None:
        cfg.setdefault('testing', {})
        cfg['testing']['dump_rescore_dir'] = args.dump_rescore_dir

    if args.dump_rscin_hist is not None:
        cfg.setdefault('testing', {})
        cfg['testing']['dump_rscin_hist'] = args.dump_rscin_hist.lower() == 'true'
    if args.dump_rscin_hist_dir is not None:
        cfg.setdefault('testing', {})
        cfg['testing']['dump_rscin_hist_dir'] = args.dump_rscin_hist_dir

    ib_cfg = cfg.setdefault('inductive_bank', {})
    if args.inductive_bank_max_images is not None:
        ib_cfg['max_images'] = int(args.inductive_bank_max_images)
    if args.inductive_bank_sample_mode is not None:
        ib_cfg['sample_mode'] = str(args.inductive_bank_sample_mode)
    if args.inductive_bank_sample_seed is not None:
        ib_cfg['sample_seed'] = int(args.inductive_bank_sample_seed)

    # Transductive overrides (optional)
    def _as_bool(v: str) -> bool:
        return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')

    td = cfg.setdefault('transductive', {})
    if args.transductive_enabled is not None:
        td['enabled'] = _as_bool(args.transductive_enabled)
    if args.transductive_num_passes is not None:
        td['num_passes'] = int(args.transductive_num_passes)
    if args.transductive_bank_ratio_mode is not None:
        td['bank_ratio_mode'] = str(args.transductive_bank_ratio_mode)
    if args.transductive_bank_ratio is not None:
        td['bank_ratio'] = float(args.transductive_bank_ratio)
    if args.transductive_bank_ratio_min is not None:
        td['bank_ratio_min'] = float(args.transductive_bank_ratio_min)
    if args.transductive_bank_ratio_max is not None:
        td['bank_ratio_max'] = float(args.transductive_bank_ratio_max)
    if args.transductive_bank_filter is not None:
        td['bank_filter'] = _as_bool(args.transductive_bank_filter)
    if args.transductive_retrieval_topk is not None:
        td['retrieval_topk'] = int(args.transductive_retrieval_topk)
    if args.transductive_score_pool is not None:
        td['score_pool'] = str(args.transductive_score_pool)
    if args.transductive_msm_topmin_min is not None:
        td['msm_topmin_min'] = float(args.transductive_msm_topmin_min)
    if args.transductive_msm_topmin_max is not None:
        td['msm_topmin_max'] = float(args.transductive_msm_topmin_max)
    if args.transductive_safe_fusion is not None:
        td['safe_fusion'] = _as_bool(args.transductive_safe_fusion)
    if args.transductive_safe_fusion_weight is not None:
        td['safe_fusion_weight'] = float(args.transductive_safe_fusion_weight)
    if args.transductive_adaptive_gate is not None:
        td['adaptive_gate'] = _as_bool(args.transductive_adaptive_gate)
    if args.transductive_gate_threshold is not None:
        td['gate_threshold'] = float(args.transductive_gate_threshold)
    if args.transductive_apply_to_pixel is not None:
        td['apply_to_pixel'] = _as_bool(args.transductive_apply_to_pixel)
    if args.transductive_knee_strength_min is not None:
        td['knee_strength_min'] = float(args.transductive_knee_strength_min)
    if args.transductive_knee_tail_frac is not None:
        td['knee_tail_frac'] = float(args.transductive_knee_tail_frac)
    return cfg

if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)

    # Record lightweight metadata for reproducibility (also saved into results.xlsx).
    cfg.setdefault("meta", {})
    if not isinstance(cfg["meta"], dict):
        cfg["meta"] = {}
    meta = cfg["meta"]
    meta.setdefault("config_path", os.path.abspath(args.config))
    meta.setdefault("cmd", " ".join(sys.argv))
    meta.setdefault("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    meta.setdefault("python", sys.version.replace("\n", " "))
    meta.setdefault("conda_env", os.environ.get("CONDA_DEFAULT_ENV", ""))
    meta.setdefault("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    try:
        meta.setdefault(
            "git_commit",
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        )
    except Exception:
        pass

    print(cfg)
    seed = 42
    model = MuSc(cfg, seed=seed)
    model.main()



