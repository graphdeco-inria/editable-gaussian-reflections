import json
import pandas as pd
import copy

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

metrics_arrow = { "psnr": "up", "ssim": "up", "lpips": "down"}

method_ordering = [
    "3dgs",  
    "2dgs",
    "r3dg",
    "gaussian_shader", 
    "3dgs_dr", 
    "ref_gaussian", 
    "ours"
]
pass_ordering = ["normal", "diffuse", "glossy", "render"]

pass_renaming = {
    "normal": "{\scriptsize Norm.}",
    "render": "{\scriptsize Final}",
    "diffuse": "{\scriptsize Diff.}",
    "glossy": "{\scriptsize Spec.}"
}

method_renaming = {
    "2dgs": "2DGS",
    "3dgs": "3DGS",
    "r3dg": "Relightable 3DGS",
    "3dgs_dr": "3DGS-DR",
    "gaussian_shader": "Gaussian Shader",
    "ref_gaussian": "Reflective GS",
    "ours": "Ours (gt+1bounce+500k)"
}

scene_renaming = {
    "shiny_kitchen": "Shiny Kitchen",
    "shiny_bedroom": "Shiny Bedroom",
    "shiny_office": "Shiny Office",
    "shiny_livingroom": "Shiny Livingroom"
}

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
        latex_file.write("\\setlength{\\tabcolsep}{5.0pt}\n")
        
        for metric in metrics_arrow.keys():
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
                .replace("\multicolumn{4}{r}", "\multicolumn{4}{r@{\hspace{6mm}}}", len(scene_renaming.keys()) - 1).replace("\multicolumn{4}{l}", "\multicolumn{4}{r}")
                .replace("{lllllllllllllllll}", "{l@{\hspace{6mm}}rrrr@{\hspace{6mm}}rrrr@{\hspace{6mm}}rrrr@{\hspace{6mm}}rrrr}").replace("Scene", "").replace("Render Pass", "")
            )

            
            latex_file.write("\n\n")
            
        # potential other comaprions: beta splatting, interreflective 3dgs, envgs, 
        
        latex_file.write("\\subsubsection*{Notes}\n")
        latex_file.write("TONEMAPPING IS INCORRECT IN ALL EVALS --- only the final render scores are valid. Need to figure out what to do with tonemapping.\\\\")
        latex_file.write("Our method uses ground truths for all targets (but fits the values to the guassians), 1 bounce of reflection, and 500k gaussians.\\\\")
        latex_file.write("R3DG diffuse maps seem wrong/way overexposed.\\\\")
        latex_file.write("Normal + Depth maps should be given to R3DG since its what they do for tanks and temples, however they don't provide the method for this. Best is we give them the same maps we use.\\\\")

        latex_file.write("\\end{document}\n")
