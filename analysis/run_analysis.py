"""
Public evaluation script for the voice-avatar mismatch study.

This script reads the released result filenames:
- task1_photorealistic.csv
- task2_stylized.csv
- task2_pixel_art.csv
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Stake level classification
STAKE_MAP = {
    'BR': 'high',
    'SD': 'high',
    'SV': 'high',
    'CS': 'low',
    'CZ': 'low',
}


def find_csv_file(model_dir: Path, prefix: str) -> Optional[Path]:
    """Find CSV file matching prefix, with or without suffix."""
    # Try exact match first
    exact_path = model_dir / f"{prefix}.csv"
    if exact_path.exists():
        return exact_path

    # Try with suffix pattern (e.g., task2_two_avatars_full_modelname.csv)
    for path in model_dir.glob(f"{prefix}*.csv"):
        return path

    return None


def find_any_csv_file(model_dir: Path, prefixes: list[str]) -> Optional[Path]:
    for prefix in prefixes:
        path = find_csv_file(model_dir, prefix)
        if path:
            return path
    return None


def load_model_data(model_name: str) -> dict[str, pd.DataFrame]:
    """Load all public-condition data for a model."""
    model_dir = RESULTS_DIR / model_name
    data = {}

    public_datasets = {
        'photorealistic': ['task1_photorealistic'],
        'stylized': ['task2_stylized'],
        'pixel_art': ['task2_pixel_art'],
    }

    for dataset_name, prefixes in public_datasets.items():
        dataset_path = find_any_csv_file(model_dir, prefixes)
        if dataset_path:
            data[dataset_name] = pd.read_csv(dataset_path)
            if 'style' not in data[dataset_name].columns:
                data[dataset_name]['style'] = dataset_name

    return data


def preprocess_data(df: pd.DataFrame, for_cross_style: bool = False) -> pd.DataFrame:
    """Add derived variables for analysis.

    Args:
        df: Raw data DataFrame
        for_cross_style: If True, add style prefix to image_pair for cross-style analysis (48 levels)
                        If False, use order-independent image_pair (16 levels)
    """
    df = df.copy()

    # Filter only successful trials
    df = df[df['status'] == 'ok'].copy()

    # Binary predictors
    df['voice_male'] = (df['voice_gender'] == 'male').astype(int)
    df['male_first'] = (df['order'] == 'MF').astype(int)

    # Binary outcomes
    df['follow_male'] = (df['follow_gender'] == 'male').astype(int)
    df['follow_first'] = (df['follow'] == 'avatar_1').astype(int)

    # Voice-avatar match (DV for 1-2)
    df['voice_avatar_match'] = (df['voice_gender'] == df['follow_gender']).astype(int)

    # Match avatar position (IV for 1-2): matching avatar가 첫 번째 위치인지
    # voice_male=1 & male_first=1 → match avatar is first
    # voice_male=0 & male_first=0 → match avatar is first (female voice, female first)
    df['match_avatar_first'] = ((df['voice_male'] == df['male_first'])).astype(int)

    # Genre and stake level
    df['genre'] = df['scenario'].str[:2]
    df['stake_level'] = df['genre'].map(STAKE_MAP)

    # Image pair identifier (order-independent: 16 levels)
    # Use sorted IDs to make it position-independent
    df['avatar_1_id_str'] = df['avatar_1_id'].astype(str)
    df['avatar_2_id_str'] = df['avatar_2_id'].astype(str)
    df['image_pair'] = df.apply(
        lambda row: f"{min(row['avatar_1_id_str'], row['avatar_2_id_str'])}_{max(row['avatar_1_id_str'], row['avatar_2_id_str'])}",
        axis=1
    )

    # For cross-style analysis, add style prefix (48 levels)
    if for_cross_style:
        df['image_pair'] = df['style'] + '_' + df['image_pair']

    # Clean up temporary columns
    df = df.drop(columns=['avatar_1_id_str', 'avatar_2_id_str'])

    return df


def descriptive_stats(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Calculate descriptive statistics."""
    stats = []

    # Overall
    stats.append({
        'group': 'Overall',
        'n': len(df),
        'follow_male_rate': df['follow_male'].mean(),
        'follow_male_se': df['follow_male'].std() / np.sqrt(len(df)),
        'follow_first_rate': df['follow_first'].mean(),
        'voice_avatar_match_rate': df['voice_avatar_match'].mean(),
    })

    # By voice gender
    for vg in ['male', 'female']:
        subset = df[df['voice_gender'] == vg]
        if len(subset) > 0:
            stats.append({
                'group': f'Voice: {vg}',
                'n': len(subset),
                'follow_male_rate': subset['follow_male'].mean(),
                'follow_male_se': subset['follow_male'].std() / np.sqrt(len(subset)),
                'follow_first_rate': subset['follow_first'].mean(),
                'voice_avatar_match_rate': subset['voice_avatar_match'].mean(),
            })

    # By stake level
    for sl in ['high', 'low']:
        subset = df[df['stake_level'] == sl]
        if len(subset) > 0:
            stats.append({
                'group': f'Stake: {sl}',
                'n': len(subset),
                'follow_male_rate': subset['follow_male'].mean(),
                'follow_male_se': subset['follow_male'].std() / np.sqrt(len(subset)),
                'follow_first_rate': subset['follow_first'].mean(),
                'voice_avatar_match_rate': subset['voice_avatar_match'].mean(),
            })

    # By order
    for order in ['MF', 'FM']:
        subset = df[df['order'] == order]
        if len(subset) > 0:
            stats.append({
                'group': f'Order: {order}',
                'n': len(subset),
                'follow_male_rate': subset['follow_male'].mean(),
                'follow_male_se': subset['follow_male'].std() / np.sqrt(len(subset)),
                'follow_first_rate': subset['follow_first'].mean(),
                'voice_avatar_match_rate': subset['voice_avatar_match'].mean(),
            })

    # By voice gender × stake level
    for vg in ['male', 'female']:
        for sl in ['high', 'low']:
            subset = df[(df['voice_gender'] == vg) & (df['stake_level'] == sl)]
            if len(subset) > 0:
                stats.append({
                    'group': f'Voice:{vg} × Stake:{sl}',
                    'n': len(subset),
                    'follow_male_rate': subset['follow_male'].mean(),
                    'follow_male_se': subset['follow_male'].std() / np.sqrt(len(subset)),
                    'follow_first_rate': subset['follow_first'].mean(),
                    'voice_avatar_match_rate': subset['voice_avatar_match'].mean(),
                })

    stats_df = pd.DataFrame(stats)
    stats_df['dataset'] = name

    return stats_df


def generate_r_script(data_path: Path, output_prefix: str, analysis_name: str) -> str:
    """Generate R script for GLMM analysis."""

    r_script = f'''
# Voice-Avatar Gender Mismatch Analysis: {analysis_name}
# Auto-generated R script

suppressPackageStartupMessages({{
    library(lme4)
    library(lmerTest)
}})

# Load data
df <- read.csv("{data_path}")

# Convert to factors
df$stake_level <- factor(df$stake_level, levels = c("low", "high"))
df$voice <- factor(df$voice)
df$image_pair <- factor(df$image_pair)
df$scenario <- factor(df$scenario)

cat("\\n========================================\\n")
cat("Analysis: {analysis_name}\\n")
cat("========================================\\n")
cat("N trials:", nrow(df), "\\n")
cat("N voices:", length(unique(df$voice)), "\\n")
cat("N scenarios:", length(unique(df$scenario)), "\\n")
cat("N image pairs:", length(unique(df$image_pair)), "\\n\\n")

results <- list()

# ============================================
# Analysis 1-1: Overall Gender Bias
# DV: follow_male, IV: voice_male, male_first
# ============================================
cat("\\n--- Analysis 1: Voice-Based Avatar Selection Bias (Main) ---\\n")

tryCatch({{
    model_1_1 <- glmer(
        follow_male ~ voice_male + male_first +
        (1|voice) + (1|image_pair) + (1|scenario),
        data = df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
    )

    cat("\\nFixed Effects:\\n")
    print(summary(model_1_1)$coefficients)

    cat("\\nOdds Ratios:\\n")
    or <- exp(fixef(model_1_1))
    print(or)

    # Interpretation
    cat("\\nInterpretation:\\n")
    cat("- Intercept: prob(follow_male) when voice=female, male_first=0\\n")
    cat("- voice_male: effect of male voice on following male avatar\\n")
    cat("- male_first: position bias (male in first position)\\n")

    results$model_1_1 <- list(
        converged = TRUE,
        fixed_effects = as.data.frame(summary(model_1_1)$coefficients),
        odds_ratios = or
    )
}}, error = function(e) {{
    cat("Error in 1-1:", e$message, "\\n")
    results$model_1_1 <<- list(converged = FALSE, error = e$message)
}})

# ============================================
# Analysis 1-2: Voice-Avatar Match Effect
# DV: voice_avatar_match, IV: match_avatar_first
# ============================================
cat("\\n--- Analysis 2: Matching Tendency (Confirmatory) ---\\n")

tryCatch({{
    model_1_2 <- glmer(
        voice_avatar_match ~ match_avatar_first +
        (1|voice) + (1|image_pair) + (1|scenario),
        data = df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
    )

    cat("\\nFixed Effects:\\n")
    print(summary(model_1_2)$coefficients)

    cat("\\nOdds Ratios:\\n")
    or <- exp(fixef(model_1_2))
    print(or)

    # Interpretation
    intercept_logit <- fixef(model_1_2)["(Intercept)"]
    intercept_prob <- plogis(intercept_logit)
    cat("\\nInterpretation:\\n")
    cat("- Intercept (logit):", round(intercept_logit, 3), "\\n")
    cat("- Intercept (prob):", round(intercept_prob, 3), "\\n")
    cat("- If intercept prob > 0.5, there is voice-avatar match tendency\\n")
    cat("- match_avatar_first: position effect on matching\\n")

    results$model_1_2 <- list(
        converged = TRUE,
        fixed_effects = as.data.frame(summary(model_1_2)$coefficients),
        odds_ratios = or,
        intercept_prob = intercept_prob
    )
}}, error = function(e) {{
    cat("Error in 1-2:", e$message, "\\n")
    results$model_1_2 <<- list(converged = FALSE, error = e$message)
}})

# ============================================
# Analysis 1-3: Stake Level Effect
# DV: follow_male, IV: voice_male * stake_level, male_first
# ============================================
cat("\\n--- Analysis 3: Stake Level Moderation ---\\n")

tryCatch({{
    model_1_3 <- glmer(
        follow_male ~ voice_male * stake_level + male_first +
        (1|voice) + (1|image_pair) + (1|scenario),
        data = df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
    )

    cat("\\nFixed Effects:\\n")
    print(summary(model_1_3)$coefficients)

    cat("\\nOdds Ratios:\\n")
    or <- exp(fixef(model_1_3))
    print(or)

    # Interpretation
    cat("\\nInterpretation:\\n")
    cat("- stake_levelhigh: effect of high stake (vs low) on following male\\n")
    cat("- voice_male:stake_levelhigh: interaction - does voice effect differ by stake?\\n")

    results$model_1_3 <- list(
        converged = TRUE,
        fixed_effects = as.data.frame(summary(model_1_3)$coefficients),
        odds_ratios = or
    )
}}, error = function(e) {{
    cat("Error in 1-3:", e$message, "\\n")
    results$model_1_3 <<- list(converged = FALSE, error = e$message)
}})

# ============================================
# Analysis 1-4: Primacy Effect
# DV: follow_first, IV: intercept only
# ============================================
cat("\\n--- Analysis 4: Position Effect ---\\n")

tryCatch({{
    model_1_4 <- glmer(
        follow_first ~ 1 +
        (1|voice) + (1|image_pair) + (1|scenario),
        data = df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
    )

    cat("\\nFixed Effects:\\n")
    print(summary(model_1_4)$coefficients)

    cat("\\nOdds Ratios:\\n")
    or <- exp(fixef(model_1_4))
    print(or)

    # Baseline probability
    intercept_logit <- fixef(model_1_4)["(Intercept)"]
    p_first <- plogis(intercept_logit)
    cat("\\nInterpretation:\\n")
    cat("- Intercept (logit):", round(intercept_logit, 3), "\\n")
    cat("- Baseline prob of following first avatar:", round(p_first, 3), "\\n")
    cat("- If prob > 0.5, there is primacy bias\\n")

    results$model_1_4 <- list(
        converged = TRUE,
        fixed_effects = as.data.frame(summary(model_1_4)$coefficients),
        odds_ratios = or,
        baseline_prob = p_first
    )
}}, error = function(e) {{
    cat("Error in 1-4:", e$message, "\\n")
    results$model_1_4 <<- list(converged = FALSE, error = e$message)
}})

# Save results as JSON
results_json <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(results_json, "{OUTPUT_DIR}/{output_prefix}_results.json")
cat("\\nResults saved to {OUTPUT_DIR}/{output_prefix}_results.json\\n")
'''

    return r_script


def generate_style_comparison_r_script(data_path: Path, output_prefix: str) -> str:
    """Generate R script for cross-style comparison (Analysis 5)."""

    r_script = f'''
# Voice-Avatar Gender Mismatch Analysis: Analysis 5 - Style Comparison
# Auto-generated R script
# Note: image_pair has style prefix (e.g., "photorealistic_1_2") for 48 levels

suppressPackageStartupMessages({{
    library(lme4)
    library(lmerTest)
    if(!require(emmeans)) install.packages("emmeans", repos="https://cran.r-project.org", quiet=TRUE)
    library(emmeans)
}})

# Load data
df <- read.csv("{data_path}")

# Convert to factors
df$style <- factor(df$style, levels = c("photorealistic", "stylized", "pixel_art"))
df$voice <- factor(df$voice)
df$image_pair <- factor(df$image_pair)
df$scenario <- factor(df$scenario)

cat("\\n========================================\\n")
cat("Analysis 5: Style Comparison\\n")
cat("========================================\\n")
cat("N trials:", nrow(df), "\\n")
cat("N voices:", length(unique(df$voice)), "\\n")
cat("N scenarios:", length(unique(df$scenario)), "\\n")
cat("N image pairs (style-prefixed):", length(unique(df$image_pair)), "\\n")
cat("Styles:", paste(unique(as.character(df$style)), collapse=", "), "\\n\\n")

results <- list()

tryCatch({{
    model_5 <- glmer(
        follow_male ~ voice_male * style + male_first +
        (1|voice) + (1|image_pair) + (1|scenario),
        data = df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
    )

    cat("\\n--- Fixed Effects ---\\n")
    print(summary(model_5)$coefficients)

    cat("\\n--- Odds Ratios ---\\n")
    or <- exp(fixef(model_5))
    print(or)

    # Interpretation
    cat("\\nInterpretation:\\n")
    cat("- Intercept: prob(follow_male) for photorealistic style, female voice, male_first=0\\n")
    cat("- stylestylized: effect of stylized (vs photorealistic) on following male\\n")
    cat("- stylepixel_art: effect of pixel_art (vs photorealistic) on following male\\n")
    cat("- voice_male:style*: does voice effect differ by style?\\n")

    # Check if interaction is significant
    coef_table <- summary(model_5)$coefficients
    interaction_terms <- grep("voice_male:style", rownames(coef_table))
    interaction_pvals <- coef_table[interaction_terms, "Pr(>|z|)"]
    interaction_sig <- any(interaction_pvals < 0.05)

    cat("\\n--- Post-hoc Analysis ---\\n")

    # 1. Style pairwise comparison
    cat("\\n[Style Pairwise Comparison (Tukey)]\\n")
    emm_style <- emmeans(model_5, ~ style)
    style_pairs <- pairs(emm_style, adjust = "tukey")
    print(style_pairs)

    # 2. Simple effects (if interaction is significant)
    if(interaction_sig) {{
        cat("\\n[Simple Effects: Voice effect within each style]\\n")
        cat("(Interaction is significant, examining voice effect per style)\\n\\n")
        emm_voice_style <- emmeans(model_5, ~ voice_male | style)
        simple_effects <- pairs(emm_voice_style)
        print(simple_effects)
    }} else {{
        cat("\\n[Simple Effects: Not computed]\\n")
        cat("(Interaction is NOT significant, simple effects not needed)\\n")
    }}

    results$model_5 <- list(
        converged = TRUE,
        fixed_effects = as.data.frame(summary(model_5)$coefficients),
        odds_ratios = or,
        interaction_significant = interaction_sig
    )
}}, error = function(e) {{
    cat("Error in Analysis 5:", e$message, "\\n")
    results$model_5 <- list(converged = FALSE, error = e$message)
}})

# Save results
results_json <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(results_json, "{OUTPUT_DIR}/{output_prefix}_style_comparison_results.json")
cat("\\nResults saved to {OUTPUT_DIR}/{output_prefix}_style_comparison_results.json\\n")
'''

    return r_script


def run_r_script(script_path: Path) -> tuple[bool, str]:
    """Execute R script and return success status and output."""
    try:
        result = subprocess.run(
            ['Rscript', str(script_path)],
            capture_output=True,
            text=True,
            timeout=600
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        return result.returncode == 0, output
    except FileNotFoundError:
        return False, "R not found. Please install R."
    except subprocess.TimeoutExpired:
        return False, "R script timed out."


def run_analysis(model_name: str, run_r: bool = True):
    """Run complete analysis for a model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    # Load data
    data = load_model_data(model_name)

    if not data:
        print(f"No data found for {model_name}")
        return None

    all_stats = []

    # Process each task
    for task_name, df in data.items():
        print(f"\n--- {task_name} ---")
        print(f"Raw trials: {len(df)}")

        # Preprocess
        df = preprocess_data(df)
        print(f"Valid trials (status=ok): {len(df)}")

        if len(df) == 0:
            print("No valid trials, skipping...")
            continue

        # Descriptive stats
        stats = descriptive_stats(df, f"{model_name}_{task_name}")
        all_stats.append(stats)

        print(f"\nDescriptive Statistics:")
        print(stats[['group', 'n', 'follow_male_rate', 'follow_male_se']].to_string(index=False))

        # Save preprocessed data for R
        data_path = OUTPUT_DIR / f"{model_name}_{task_name}_data.csv"
        df.to_csv(data_path, index=False)
        print(f"\nData saved: {data_path}")

        # Generate R script
        output_prefix = f"{model_name}_{task_name}"
        r_script = generate_r_script(data_path, output_prefix, f"{model_name} - {task_name}")

        r_script_path = OUTPUT_DIR / f"{output_prefix}_analysis.R"
        with open(r_script_path, 'w') as f:
            f.write(r_script)
        print(f"R script saved: {r_script_path}")

        # Run R script
        if run_r:
            print("\nRunning R analysis...")
            success, output = run_r_script(r_script_path)
            print(output)

            if success:
                # Load results
                results_path = OUTPUT_DIR / f"{output_prefix}_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        results = json.load(f)
                    print(f"\nResults loaded from {results_path}")

    # Cross-style comparison
    if len(data) > 1 and run_r:
        print(f"\n{'='*60}")
        print("Cross-Style Comparison (Analysis 2-2)")
        print(f"{'='*60}")

        # Combine all data with style-prefixed image_pair for cross-style analysis
        combined_dfs = []
        for task_name, df in data.items():
            # Use for_cross_style=True to add style prefix to image_pair
            df_proc = preprocess_data(df, for_cross_style=True)
            combined_dfs.append(df_proc)

        combined_df = pd.concat(combined_dfs, ignore_index=True)

        # Find common scenarios
        common_scenarios = None
        for df_part in combined_dfs:
            scenarios = set(df_part['scenario'].unique())
            if common_scenarios is None:
                common_scenarios = scenarios
            else:
                common_scenarios &= scenarios

        if common_scenarios:
            combined_df = combined_df[combined_df['scenario'].isin(common_scenarios)]
            print(f"Common scenarios: {len(common_scenarios)}")
            print(f"Combined trials: {len(combined_df)}")
            print(f"Image pairs (style-prefixed): {combined_df['image_pair'].nunique()}")

            # Save combined data
            combined_data_path = OUTPUT_DIR / f"{model_name}_combined_data.csv"
            combined_df.to_csv(combined_data_path, index=False)

            # Generate and run style comparison script
            r_script_style = generate_style_comparison_r_script(combined_data_path, model_name)
            r_script_style_path = OUTPUT_DIR / f"{model_name}_style_comparison.R"
            with open(r_script_style_path, 'w') as f:
                f.write(r_script_style)

            print(f"\nRunning style comparison analysis...")
            success, output = run_r_script(r_script_style_path)
            print(output)

    # Save combined descriptive stats
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        stats_path = OUTPUT_DIR / f"{model_name}_descriptive_stats.csv"
        combined_stats.to_csv(stats_path, index=False)
        print(f"\nDescriptive stats saved: {stats_path}")
        return combined_stats

    return None


def get_available_models() -> list[str]:
    """Get list of models with results."""
    models = []
    for path in RESULTS_DIR.iterdir():
        if path.is_dir() and not path.name.startswith('.') and not path.name.startswith('backup'):
            csvs = list(path.glob("*.csv"))
            if csvs:
                models.append(path.name)
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description="Voice-Avatar Gender Mismatch Analysis")
    parser.add_argument('--model', type=str, default='all',
                       help='Model name or "all" for all models')
    parser.add_argument('--no-r', action='store_true',
                       help='Skip R analysis (only descriptive stats)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')

    args = parser.parse_args()

    available_models = get_available_models()

    if args.list_models:
        print("Available models:")
        for m in available_models:
            print(f"  - {m}")
        return

    # Check R installation
    if not args.no_r:
        try:
            result = subprocess.run(['R', '--version'], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("Warning: R not found. Running with --no-r")
                args.no_r = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: R not found. Running with --no-r")
            args.no_r = True

        # Check jsonlite
        if not args.no_r:
            result = subprocess.run(
                ['R', '-e', 'if(!require(jsonlite)) install.packages("jsonlite", repos="https://cran.r-project.org", quiet=TRUE)'],
                capture_output=True,
                timeout=60
            )

    if args.model == 'all':
        models_to_analyze = available_models
    else:
        models_to_analyze = [args.model]

    all_results = []

    for model in models_to_analyze:
        if model not in available_models:
            print(f"Warning: {model} not found in results")
            continue

        stats = run_analysis(model, run_r=not args.no_r)
        if stats is not None:
            all_results.append(stats)

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*60}")

        combined = pd.concat(all_results, ignore_index=True)

        # Overall stats by model
        overall = combined[combined['group'] == 'Overall'].copy()
        print("\nOverall Follow Male Rate by Model/Task:")
        print(overall[['dataset', 'n', 'follow_male_rate', 'follow_male_se']].to_string(index=False))

        combined.to_csv(OUTPUT_DIR / "all_models_descriptive_stats.csv", index=False)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
