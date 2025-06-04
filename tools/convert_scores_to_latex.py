import json
import pandas as pd
import copy

metrics_arrow = { "psnr": "up", "ssim": "up", "lpips": "down", "num_gaussians": "down", "time": "down", "flip": "low" }

method_ordering = [
    "3dgs",  
    "2dgs",
    "gaussian_shader", 
    "3dgs_dr", 
    "ref_gaussian",
    "envgs_network", 
    "ours_network",
    "envgs_gt",
    "ours"
]
pass_ordering = [ "diffuse", "glossy", "render" ]

pass_renaming = {
    "normal": "\makebox[0pt][l]{\scriptsize Normal}",
    "render": "\makebox[0pt][l]{\scriptsize Final}",
    "diffuse": "\makebox[0pt][l]{\scriptsize Diffuse}",
    "glossy": "\makebox[0pt][l]{\scriptsize Residual}"
}

method_renaming = {
    "2dgs": "2DGS",
    "3dgs": "3DGS",
    "r3dg": "Relightable 3DGS",
    "3dgs_dr": "3DGS-DR",
    "gaussian_shader": "Gaussian Shader",
    "ref_gaussian": "Reflective GS",
    "envgs_network": "EnvGS (network)",
    "envgs_gt": "EnvGS (optimal)",
    "ours_network": "Ours (network)",
    "ours": "Ours (optimal)",
    "ground_truth": "Ground Truth"
}

method_renaming_short = {
    "2dgs": "2DGS",
    "3dgs": "3DGS",
    "r3dg": "R3DGS",
    "3dgs_dr": "3DGS-DR",
    "gaussian_shader": "GShader",
    "ref_gaussian": "ReflGS",
    "envgs_network": "EnvGS\_{net}",
    "envgs_gt": "EnvGS\_{opt}",
    "ours_network": "Ours\_{net}",
    "ours": "Ours\_{opt}",
    "ground_truth": "G.T."
}

scene_renaming = {
    "shiny_kitchen": "Shiny Kitchen",
    "shiny_bedroom": "Shiny Bedroom",
    "shiny_office": "Shiny Office",
    "shiny_livingroom": "Shiny Livingroom"
}


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def process_data(data):
    records = []
    for scene, methods in data.items():
        for method, render_passes in methods.items():
            for render_pass, metrics in render_passes.items():
                for metric, value in metrics.items():
                    records.append({
                        'Scene': scene,
                        'Method': method,
                        'Render Pass': render_pass,
                        'Metric': metric,
                        'Value': value
                    })
    return records

if __name__ == "__main__":
    file_path = 'scores.json' 
    data = load_json(file_path)
    records = process_data(data)
    
    with open('scores.tex', 'w') as latex_file:
        latex_file.write("\\documentclass{article}\n")
        latex_file.write("\\usepackage[landscape]{geometry}\n")
        latex_file.write("\\usepackage{tabularx}\n")
        latex_file.write("\\usepackage{booktabs}\n")
        latex_file.write("\\usepackage[table]{xcolor}\n")
        latex_file.write("\\geometry{left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm}\n")
        latex_file.write("\\begin{document}\n")
        latex_file.write("\\setlength{\\tabcolsep}{8.0pt}\n")
        
        for metric in ["psnr", "ssim"]: # "psnr", "ssim", "lpips"
            # Filter records to only include the current metric and exclude 'normal' render pass
            filtered_records = [record for record in records if record['Metric'] == metric]
            
            # Create a DataFrame from the filtered records
            df = pd.DataFrame(filtered_records)
            
            # Pivot the DataFrame to have scenes and render passes as columns
            df_pivot = df.pivot_table(index='Method', columns=['Scene', 'Render Pass'], values='Value', aggfunc='first')
            
            # Reorder the columns to have a multi-level column index with scenes as the first level
            df_pivot.columns = pd.MultiIndex.from_tuples(df_pivot.columns, names=['Scene', 'Render Pass'])

            # Reindex the DataFrame to match the order in 'methods'
            df_pivot = df_pivot.reindex(method_ordering)

            # Reorder the columns to match the order in 'pass_ordering'
            df_pivot = df_pivot.reindex(columns=pd.MultiIndex.from_product(
                [df_pivot.columns.levels[0], pass_ordering],
                names=['Scene', 'Render Pass']
            ))
            df_pivot_unrounded = df_pivot
            
            # Round values to 2 decimal places and keep trailing zeros
            if metric == "psnr":
                df_pivot = df_pivot.applymap(lambda x: (f" {x:.2f}" if x < 10 else f"{x:.2f}") if not pd.isna(x) else "---\ \ \ ")
            else:
                df_pivot = df_pivot.applymap(lambda x: (f"{x:.3f}" if x < 10 else f"{x:.3f}") if not pd.isna(x) else "---\ \ \ ")
                
            # Highlight the top scoring methods
            def highlight_max(s):
                col = df_pivot_unrounded[s.name[0]][s.name[1]]
                is_best = col == col.max() if metrics_arrow[metric] == "up" else col == col.min()
                return ['\\cellcolor{orange!25}' + str(v) if is_best[i] else str(v) for i, v in enumerate(s)]
            df_pivot = df_pivot.apply(highlight_max, axis=0)

            # Write the table to the LaTeX file
            latex_file.write(f"\\subsubsection*{{{metric.upper()}}}\n")
            # Escape column and row names
            df_pivot.columns = pd.MultiIndex.from_tuples([(
                scene_renaming[scene],
                pass_renaming[render_pass]
            ) for scene, render_pass in df_pivot.columns], names=['Scene', 'Render Pass'])
            df_pivot.index = [method_renaming[method] for method in df_pivot.index]

            # Write the table to the LaTeX file without escaping cell values
            latex_file.write(
                df_pivot.to_latex(index=True, escape=False)
                .replace("\multicolumn{3}{r}", "\multicolumn{3}{l}")
                .replace("Scene", "").replace("Render Pass", "")
            )
            latex_file.write("\n\n")
            
        latex_file.write("\\end{document}\n")
