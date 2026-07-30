"""Stable CSV schemas. Add columns only at the end to preserve compatibility."""

STAGE_COLUMNS = [
    "experiment", "scene", "run_id", "stage", "parent_stage", "started_at",
    "ended_at", "wall_seconds", "cpu_seconds", "status", "rss_before_bytes",
    "rss_after_bytes", "system_available_before_bytes", "system_available_after_bytes",
    "peak_sampled_rss_bytes", "input_points", "output_gaussians", "iterations",
    "frames", "layer_index", "exception_type", "exception_message",
    "seconds_per_iteration", "seconds_per_frame",
    "seconds_per_million_input_points", "seconds_per_million_output_gaussians",
]
RESOURCE_COLUMNS = [
    "experiment", "scene", "run_id", "timestamp", "elapsed_seconds", "stage",
    "process_rss_bytes", "process_vms_bytes", "system_total_bytes",
    "system_available_bytes", "system_used_bytes", "system_used_percent",
    "swap_total_bytes", "swap_used_bytes", "process_cpu_percent",
]
LAYER_COLUMNS = [
    "experiment", "scene", "run_id", "variant", "layer_index", "semantic_label",
    "instance_ids", "confidence_count", "confidence_mean", "confidence_min",
    "confidence_max", "mask_area_pixels", "mask_coverage_percent",
    "connected_components", "projected_3d_points", "training_frames",
    "total_supervised_pixels", "mean_supervised_pixels_per_frame",
    "training_iterations", "training_time_seconds", "initial_gaussians",
    "final_gaussians", "ply_size_bytes", "percent_final_scene_gaussians",
    "status", "reason",
]
SEGMENTATION_COLUMNS = [
    "experiment", "scene", "run_id", "target", "metric_scope", "iou", "dice",
    "precision", "recall", "boundary_fscore", "false_positive_pixels",
    "false_negative_pixels", "thin_structure_iou", "coverage_percent",
    "background_percent", "unassigned_percent", "overlap_before_pixels",
    "overlap_after_pixels", "seam_crossing", "source",
]
RECONSTRUCTION_COLUMNS = [
    "experiment", "scene", "run_id", "variant", "view_id", "split", "theta_deg",
    "phi_deg", "width", "height", "psnr_db", "ssim", "lpips", "mae",
    "foreground_psnr_db", "foreground_ssim", "background_psnr_db",
    "background_ssim", "render_seconds", "status", "note",
]
RENDERING_COLUMNS = [
    "experiment", "scene", "run_id", "variant", "target", "width", "height",
    "warmup_frames", "measured_frames", "cold_start_seconds", "mean_ms",
    "median_ms", "p90_ms", "p95_ms", "average_fps", "minimum_fps",
    "gaussian_count", "megapixels_per_second", "status", "reason",
]
EDITING_COLUMNS = [
    "experiment", "scene", "run_id", "variant", "target_type", "target_id",
    "inside_changed_percent", "outside_changed_percent", "outside_mae",
    "outside_lpips", "edit_leakage_ratio", "edit_locality_score",
    "removed_gaussians", "removed_gaussians_percent", "creation_seconds",
    "retraining_required", "edited_size_bytes", "status", "reason",
]
MOOD_COLUMNS = [
    "experiment", "scene", "run_id", "day_variant", "mood_variant",
    "correspondence_compatible", "day_gaussians", "mood_gaussians",
    "gaussian_count_difference", "position_mean_abs", "position_max_abs",
    "scale_mean_abs", "scale_max_abs", "rotation_mean_abs", "rotation_max_abs",
    "opacity_mean_abs", "opacity_max_abs", "label_difference_count",
    "sh_mean_abs", "sh_max_abs", "nonappearance_changed_percent",
    "analytic_fit_seconds", "refinement_seconds", "mood_ply_size_bytes",
    "target_erp_psnr_db", "target_erp_ssim", "circular_seam_mae", "status", "reason",
]
