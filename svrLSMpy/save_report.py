from nilearn.plotting import view_img
from pathlib import Path
import pandas as pd
import nibabel as nib
import base64

from modules.plot_mosaics import get_axial_slices, get_coronal_slices, get_sagittal_slices, save_axial_mosaic, save_coronal_mosaic, save_sagittal_mosaic

def save_report(output_file, svr_params, behaviour_name, n_permutations, alpha, zmap_range, zmap, min_patient_count,num_patients, num_slices, nifti_zmap, zmap_atlas_output_dir, time_taken, num_lesions, mean_lesion_volume, n_clusters=5):
    """
    Save a comprehensive report of the LSM analysis, including parameters, significant voxels, and visualization.
    """
    print("Saving report...")

    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    atlas_reader_output_folder = Path(zmap_atlas_output_dir)
    cluster_csv_path = atlas_reader_output_folder / "atlasreader_clusters.csv"
    fileExists = False
    try:
        cluster_df = pd.read_csv(cluster_csv_path)
        print("File atlasreader_clusters.csv loaded successfully!")
        if (cluster_df.shape[0] < 5):
            n_clusters = int(cluster_df.shape[0])

        top_clusters = cluster_df.nlargest(n_clusters, 'volume_mm')  # Get top 5 clusters by volume

        overview_image_path = atlas_reader_output_folder / "atlasreader.png"
        cluster_image_paths = [atlas_reader_output_folder / f"atlasreader_cluster0{i}.png" for i in
                               range(1, n_clusters + 1)]


        overview_image_base64 = encode_image(overview_image_path)
        cluster_images_base64 = [encode_image(path) for path in cluster_image_paths]

        fileExists = True
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print("Error: The file 'atlasreader_clusters.csv' was not found.")
    except Exception as e:
        # Catch any other exceptions
        print(f"An unexpected error occurred: {e}")

    #significant_voxels = np.abs(zmap) > np.percentile(np.abs(zmap), 100 * (1 - alpha))

    output_folder = Path(Path(output_file).parent)

    mosaic_output_folder = output_folder / "mosaics"
    Path(mosaic_output_folder).mkdir(parents=True, exist_ok=True)

    zmap_threshold_output_folder = output_folder / "thresholded_zmaps"

    html_view = view_img(output_folder/"zmap.nii.gz",threshold=1.7,black_bg=False,cmap="jet")

    lesion_overlap_path = output_folder / "lesion_overlap.nii.gz"
    lesion_overlap_filtered_path = output_folder / "lesion_overlap_filtered.nii.gz"
    svr_beta_map_path = output_folder / "beta_map.nii.gz"
    zmap_path = output_folder / "zmap.nii.gz"
    zmap_p05_path = zmap_threshold_output_folder / "zmap_p05.nii.gz"
    zmap_p01_path = zmap_threshold_output_folder / "zmap_p01.nii.gz"
    zmap_p005_path = zmap_threshold_output_folder / "zmap_p005.nii.gz"
    zmap_p001_path = zmap_threshold_output_folder / "zmap_p001.nii.gz"

    #AXIAL
    cut_coords = get_axial_slices(lesion_overlap_path, num_slices)

    axial_lesion_overlap_mosaic_path = mosaic_output_folder/"axial_lesion_overlap_mosaic.png"
    save_axial_mosaic(lesion_overlap_path, cut_coords, axial_lesion_overlap_mosaic_path)
    axial_lesion_overlap_mosaic = encode_image(axial_lesion_overlap_mosaic_path)

    axial_lesion_overlap_filtered_mosaic_path = mosaic_output_folder/"axial_lesion_overlap_filtered_mosaic.png"
    save_axial_mosaic(lesion_overlap_filtered_path, cut_coords, axial_lesion_overlap_filtered_mosaic_path)
    axial_lesion_overlap_filtered_mosaic = encode_image(axial_lesion_overlap_filtered_mosaic_path)


    axial_svr_beta_map_mosaic_path = mosaic_output_folder/"axial_svr_beta_map_mosaic.png"

    cut_coords = get_axial_slices(svr_beta_map_path, num_slices)

    save_axial_mosaic(svr_beta_map_path, cut_coords, axial_svr_beta_map_mosaic_path)
    axial_svr_beta_map_mosaic = encode_image(axial_svr_beta_map_mosaic_path)


    axial_zmap_mosaic_path = mosaic_output_folder/"axial_zmap_mosaic.png"
    save_axial_mosaic(zmap_path, cut_coords, axial_zmap_mosaic_path)
    axial_zmap_mosaic = encode_image(axial_zmap_mosaic_path)

    axial_zmap_p05_mosaic_path = mosaic_output_folder/"axial_zmap_p05_mosaic.png"
    save_axial_mosaic(zmap_p05_path, cut_coords, axial_zmap_p05_mosaic_path)
    axial_zmap_p05_mosaic = encode_image(axial_zmap_p05_mosaic_path)

    axial_zmap_p01_mosaic_path = mosaic_output_folder/"axial_zmap_p01_mosaic.png"
    save_axial_mosaic(zmap_p01_path, cut_coords, axial_zmap_p01_mosaic_path)
    axial_zmap_p01_mosaic = encode_image(axial_zmap_p01_mosaic_path)

    axial_zmap_p005_mosaic_path = mosaic_output_folder/"axial_zmap_p005_mosaic.png"
    save_axial_mosaic(zmap_p005_path, cut_coords, axial_zmap_p005_mosaic_path)
    axial_zmap_p005_mosaic = encode_image(axial_zmap_p005_mosaic_path)

    axial_zmap_p001_mosaic_path = mosaic_output_folder/"axial_zmap_p001_mosaic.png"
    save_axial_mosaic(zmap_p001_path, cut_coords, axial_zmap_p001_mosaic_path)
    axial_zmap_p001_mosaic = encode_image(axial_zmap_p001_mosaic_path)



    # CORONAL
    cut_coords = get_coronal_slices(lesion_overlap_path, num_slices)

    coronal_lesion_overlap_mosaic_path = mosaic_output_folder / "coronal_lesion_overlap_mosaic.png"
    save_coronal_mosaic(lesion_overlap_path, cut_coords, coronal_lesion_overlap_mosaic_path)
    coronal_lesion_overlap_mosaic = encode_image(coronal_lesion_overlap_mosaic_path)

    coronal_lesion_overlap_filtered_mosaic_path = mosaic_output_folder / "coronal_lesion_overlap_filtered_mosaic.png"
    save_coronal_mosaic(lesion_overlap_filtered_path, cut_coords, coronal_lesion_overlap_filtered_mosaic_path)
    coronal_lesion_overlap_filtered_mosaic = encode_image(coronal_lesion_overlap_filtered_mosaic_path)


    coronal_svr_beta_map_mosaic_path = mosaic_output_folder / "coronal_svr_beta_map_mosaic.png"

    cut_coords = get_coronal_slices(svr_beta_map_path, num_slices)

    save_coronal_mosaic(svr_beta_map_path, cut_coords, coronal_svr_beta_map_mosaic_path)
    coronal_svr_beta_map_mosaic = encode_image(coronal_svr_beta_map_mosaic_path)


    coronal_zmap_mosaic_path = mosaic_output_folder / "coronal_zmap_mosaic.png"
    save_coronal_mosaic(zmap_path, cut_coords, coronal_zmap_mosaic_path)
    coronal_zmap_mosaic = encode_image(coronal_zmap_mosaic_path)

    coronal_zmap_p05_mosaic_path = mosaic_output_folder / "coronal_zmap_p05_mosaic.png"
    save_coronal_mosaic(zmap_p05_path, cut_coords, coronal_zmap_p05_mosaic_path)
    coronal_zmap_p05_mosaic = encode_image(coronal_zmap_p05_mosaic_path)

    coronal_zmap_p01_mosaic_path = mosaic_output_folder / "coronal_zmap_p01_mosaic.png"
    save_coronal_mosaic(zmap_p01_path, cut_coords, coronal_zmap_p01_mosaic_path)
    coronal_zmap_p01_mosaic = encode_image(coronal_zmap_p01_mosaic_path)

    coronal_zmap_p005_mosaic_path = mosaic_output_folder / "coronal_zmap_p005_mosaic.png"
    save_coronal_mosaic(zmap_p005_path, cut_coords, coronal_zmap_p005_mosaic_path)
    coronal_zmap_p005_mosaic = encode_image(coronal_zmap_p005_mosaic_path)

    coronal_zmap_p001_mosaic_path = mosaic_output_folder / "coronal_zmap_p001_mosaic.png"
    save_coronal_mosaic(zmap_p001_path, cut_coords, coronal_zmap_p001_mosaic_path)
    coronal_zmap_p001_mosaic = encode_image(coronal_zmap_p001_mosaic_path)



    # SAGITTAL
    cut_coords = get_sagittal_slices(lesion_overlap_path, num_slices)

    sagittal_lesion_overlap_mosaic_path = mosaic_output_folder / "sagittal_lesion_overlap_mosaic.png"
    save_sagittal_mosaic(lesion_overlap_path, cut_coords, sagittal_lesion_overlap_mosaic_path)
    sagittal_lesion_overlap_mosaic = encode_image(sagittal_lesion_overlap_mosaic_path)

    sagittal_lesion_overlap_filtered_mosaic_path = mosaic_output_folder / "sagittal_lesion_overlap_filtered_mosaic.png"
    save_sagittal_mosaic(lesion_overlap_filtered_path, cut_coords, sagittal_lesion_overlap_filtered_mosaic_path)
    sagittal_lesion_overlap_filtered_mosaic = encode_image(sagittal_lesion_overlap_filtered_mosaic_path)


    sagittal_svr_beta_map_mosaic_path = mosaic_output_folder / "sagittal_svr_beta_map_mosaic.png"

    cut_coords = get_sagittal_slices(svr_beta_map_path, num_slices)

    save_sagittal_mosaic(svr_beta_map_path, cut_coords, sagittal_svr_beta_map_mosaic_path)
    sagittal_svr_beta_map_mosaic = encode_image(sagittal_svr_beta_map_mosaic_path)


    sagittal_zmap_mosaic_path = mosaic_output_folder / "sagittal_zmap_mosaic.png"
    save_sagittal_mosaic(zmap_path, cut_coords, sagittal_zmap_mosaic_path)
    sagittal_zmap_mosaic = encode_image(sagittal_zmap_mosaic_path)

    sagittal_zmap_p05_mosaic_path = mosaic_output_folder / "sagittal_zmap_p05_mosaic.png"
    save_sagittal_mosaic(zmap_p05_path, cut_coords, sagittal_zmap_p05_mosaic_path)
    sagittal_zmap_p05_mosaic = encode_image(sagittal_zmap_p05_mosaic_path)


    sagittal_zmap_p01_mosaic_path = mosaic_output_folder / "sagittal_zmap_p01_mosaic.png"
    save_sagittal_mosaic(zmap_p01_path, cut_coords, sagittal_zmap_p01_mosaic_path)
    sagittal_zmap_p01_mosaic = encode_image(sagittal_zmap_p01_mosaic_path)

    sagittal_zmap_p005_mosaic_path = mosaic_output_folder / "sagittal_zmap_p005_mosaic.png"
    save_sagittal_mosaic(zmap_p005_path, cut_coords, sagittal_zmap_p005_mosaic_path)
    sagittal_zmap_p005_mosaic = encode_image(sagittal_zmap_p005_mosaic_path)

    sagittal_zmap_p001_mosaic_path = mosaic_output_folder / "sagittal_zmap_p001_mosaic.png"
    save_sagittal_mosaic(zmap_p001_path, cut_coords, sagittal_zmap_p001_mosaic_path)
    sagittal_zmap_p001_mosaic = encode_image(sagittal_zmap_p001_mosaic_path)


    with open(output_file, 'w') as f:
        f.write(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVR-Based Lesion-Symptom Mapping Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7fc;
                color: #333;
                margin: 20px;
                line-height: 1.7;
            }}
            h1 {{
                color: #004d99;
                font-size: 2.5em;
                margin-bottom: 20px;
            }}
            h2 {{
                color: #004d99;
                font-size: 1.8em;
                border-bottom: 3px solid #004d99;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h3 {{
                font-size: 1.5em;
                margin-top: 20px;
                color: #2c3e50;
            }}
            p {{
                font-size: 1.1em;
                margin: 15px 0;
                color: #34495e;
            }}
            ul {{
                padding-left: 20px;
                font-size: 1.1em;
            }}
            li {{
                margin-bottom: 10px;
            }}
            .container {{
                width: 85%;
                margin: auto;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #eaf2f8;
                padding: 15px;
                border-left: 5px solid #1e88e5;
            }}
            .result-table {{
                width: 100%;
                margin-top: 20px;
                border-collapse: collapse;
            }}
            .result-table th, .result-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .result-table th {{
                background-color: #f4f7fc;
                font-weight: bold;
            }}
            .result-table td {{
                background-color: #fafafa;
            }}
            a {{
                color: #004d99;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .footer {{
                text-align: center;
                font-size: 0.9em;
                color: #777;
                margin-top: 40px;
            }}
                .image-category img {{
                display: none;
                width: 100%;
                height: auto;
            }}
                .image-category img.active {{
                display: block;
            }}
            .radio-buttons {{
                display: flex;
                gap: 15px;
                margin-top: 20px;
            }}
    
            label {{
                font-size: 16px;
                color: #333;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 120px;
                height: 40px;
                background-color: #f0f0f0; /* Default background is white */
                transition: all 0.3s ease;
                text-align: center;
                font-weight: bold;
                user-select: none; /* Prevent text selection */
            }}
    
            /* Hide the default radio buttons */
            input[type="radio"] {{
                display: none;
            }}
    
            /* When the radio button is checked, change background to light gray */
            input[type="radio"]:checked + .radio-button-block {{
                background-color: #c9c9c9; /* Light gray background when clicked */
                color: #333;
            }}
    
            /* Style for the block */
            .radio-button-block {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: all 0.3s ease;
            }}
    
            /* Hover effect */
            label:hover {{
                background-color: #dfdfdf; /* Light gray hover effect */
            }}
    

        </style>
        <script>
            function switchSvrView(viewPrefix) {{
                const groups = ['group3', 'group4', 'group5', 'group6', 'group7', 'group8'];
                groups.forEach(group => {{
                    const images = document.querySelectorAll(`.image-category img[data-group='${{group}}']`);
                    images.forEach(img => img.classList.remove('active'));
                    const selectedImage = document.querySelector(`.image-category img[data-group='${{group}}'][data-view='${{viewPrefix}}']`);
                    if (selectedImage) {{
                        selectedImage.classList.add('active');
                    }}
                }});
            }}
    
            function switchLesionView(viewPrefix) {{
                const groups = ['group1', 'group2'];
                groups.forEach(group => {{
                    const images = document.querySelectorAll(`.image-category img[data-group='${{group}}']`);
                    images.forEach(img => img.classList.remove('active'));
                    const selectedImage = document.querySelector(`.image-category img[data-group='${{group}}'][data-view='${{viewPrefix}}']`);
                    if (selectedImage) {{
                        selectedImage.classList.add('active');
                    }}
                }});
            }}
    
            // Show or hide the threshold groups based on selected threshold
            function toggleThresholdGroup(threshold) {{
                const thresholdGroups = document.querySelectorAll(`.threshold-group`);
                thresholdGroups.forEach(group => {{
                    if (group.dataset.threshold === threshold) {{
                        group.style.display = 'block';
                    }} else {{
                        group.style.display = 'none';
                    }}
                }});
            }}
    
            window.onload = function() {{
                document.querySelector('input[name="lesionSwitcher"][value="axial"]').click();
                document.querySelector('input[name="svrSwitcher"][value="axial"]').click();
                document.querySelector('input[name="thresholdSwitcher"][value="unthresholded"]').click();
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>svrLSMpy Report for {behaviour_name}</h1>

            <div class="section">
                <h2>Methodology</h2>
                <p>This report is an analysis of the relationship between lesion status and the behavioral score {behaviour_name}; performing lesion-symptom mapping using support vector regression. {num_patients} binary lesion files in MNI space were analyzed, including only voxels with at least {min_patient_count} overlapping lesions. Lesion volume was controlled using vector normalization. Covariates were regressed out of the behavioral scores. Support Vector Regression (SVR) was applied with parameters gamma = {svr_params['gamma']}, cost = {svr_params['C']}, and epsilon = {svr_params['epsilon']}, employing grid search optimization. Z maps were derived from on null distributions based on {n_permutations} permutations. The analysis was completed within {time_taken}. </p>

            </div>

            <div class="section highlight">
                <h3>Key Parameters</h3>
                <ul>
                    <li><strong>Number of patients:</strong> {num_patients}</li>
                    <li><strong>Number of permutations:</strong> {n_permutations}</li>
                    <li><strong>Alpha level:</strong> {alpha}</li>
                </ul>
            </div>

            
            <div class="section">
                <h2>Lesion Overlap</h2>
                    <!-- First Set of Radio Buttons (Lesion Switcher) -->
                    <div class="radio-buttons">
                        <label>
                            <input type="radio" name="lesionSwitcher" value="axial" onclick="switchLesionView('axial');" checked>
                            <div class="radio-button-block">Axial</div>
                        </label>
                        <label>
                            <input type="radio" name="lesionSwitcher" value="coronal" onclick="switchLesionView('coronal');">
                            <div class="radio-button-block">Coronal</div>
                        </label>
                        <label>
                            <input type="radio" name="lesionSwitcher" value="sagittal" onclick="switchLesionView('sagittal');">
                            <div class="radio-button-block">Sagittal</div>
                        </label>
                    </div>
                    <!-- Lesion Overlap Group (1st div) -->
                    <div class="image-category">
                        <h3>Unfiltered</h3>
                        <img src="data:image/png;base64,{axial_lesion_overlap_mosaic}" alt="Lesion Overlap Axial" data-group="group1" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_lesion_overlap_mosaic}" alt="Lesion Overlap Coronal" data-group="group1" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_lesion_overlap_mosaic}" alt="Lesion Overlap Sagittal" data-group="group1" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
                """)
        if min_patient_count>0:
            f.write(f"""
                    <br><br>
                
                    <!-- Lesion Overlap Filtered Group (2nd div) -->
                    <div class="image-category">
                        <h3>Filtered (Minimum {min_patient_count} patients)</h3>
                        <img src="data:image/png;base64,{axial_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Axial" data-group="group2" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Coronal" data-group="group2" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_lesion_overlap_filtered_mosaic}" alt="Lesion Overlap Filtered Sagittal" data-group="group2" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
                    """)
        f.write(f"""
            </div>
            
            <div class="section">
                <h2>SVR Parameters</h2>
                <table class="result-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
            """)
        for param, value in svr_params.items():
            f.write(f"<tr><td>{param}</td><td>{value}</td></tr>\n")

        f.write(f"""
                </table>
                <div class="image-category">
                    <h3>SVR Beta Map</h3>
                    <img src="data:image/png;base64,{axial_svr_beta_map_mosaic}" alt="SVR Beta Map Axial" data-group="group3" data-view="axial" style="width: 100%; height: auto;" class="active">
                    <img src="data:image/png;base64,{coronal_svr_beta_map_mosaic}" alt="SVR Beta Map Coronal" data-group="group3" data-view="coronal" style="width: 100%; height: auto;">
                    <img src="data:image/png;base64,{sagittal_svr_beta_map_mosaic}" alt="SVR Beta Map Sagittal" data-group="group3" data-view="sagittal" style="width: 100%; height: auto;">
                </div>
                
                <h3>Permutation Tested<h3>
                <!-- Second Set of Radio Buttons (SVR Switcher) -->
                <div class="radio-buttons">
                    <label>
                        <input type="radio" name="svrSwitcher" value="axial" onclick="switchSvrView('axial');" checked>
                        <div class="radio-button-block">Axial</div>
                    </label>
                    <label>
                        <input type="radio" name="svrSwitcher" value="coronal" onclick="switchSvrView('coronal');">
                        <div class="radio-button-block">Coronal</div>
                    </label>
                    <label>
                        <input type="radio" name="svrSwitcher" value="sagittal" onclick="switchSvrView('sagittal');">
                        <div class="radio-button-block">Sagittal</div>
                    </label>
                </div>
            
                <!-- Third Set of Radio Buttons (Threshold Switcher) -->
                <div class="radio-buttons">
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="unthresholded" onclick="toggleThresholdGroup('unthresholded')" checked>
                        <div class="radio-button-block">Unthresholded</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p05" onclick="toggleThresholdGroup('p<0.05')">
                        <div class="radio-button-block">p<0.05</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p01" onclick="toggleThresholdGroup('p<0.01')">
                        <div class="radio-button-block">p<0.01</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p005" onclick="toggleThresholdGroup('p<0.005')">
                        <div class="radio-button-block">p<0.005</div>
                    </label>
                    <label>
                        <input type="radio" name="thresholdSwitcher" value="p001" onclick="toggleThresholdGroup('p<0.001')">
                        <div class="radio-button-block">p<0.001</div>
                    </label>
                </div>
            
                <!-- Z Map Group (4th div) -->
                <div class="image-category">
                    <h3>Z Map (Thresholded)</h3>
        
                    <!-- Unthresholded Images -->
                    <div class="threshold-group" data-threshold="unthresholded" style="display: block;">
                        <img src="data:image/png;base64,{axial_zmap_mosaic}" alt="Z Map Axial" data-group="group4" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_mosaic}" alt="Z Map Coronal" data-group="group4" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_mosaic}" alt="Z Map Sagittal" data-group="group4" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
        
                    <!-- Thresholded Images -->
                    <div class="threshold-group" data-threshold="p<0.05" style="display: none;">
                        <img src="data:image/png;base64,{axial_zmap_p05_mosaic}" alt="Z Map p<0.05 Axial" data-group="group5" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_p05_mosaic}" alt="Z Map p<0.05 Coronal" data-group="group5" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_p05_mosaic}" alt="Z Map p<0.05 Sagittal" data-group="group5" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
        
                    <div class="threshold-group" data-threshold="p<0.01" style="display: none;">
                        <img src="data:image/png;base64,{axial_zmap_p01_mosaic}" alt="Z Map p<0.01 Axial" data-group="group6" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_p01_mosaic}" alt="Z Map p<0.01 Coronal" data-group="group6" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_p01_mosaic}" alt="Z Map p<0.01 Sagittal" data-group="group6" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
        
                    <div class="threshold-group" data-threshold="p<0.005" style="display: none;">
                        <img src="data:image/png;base64,{axial_zmap_p005_mosaic}" alt="Z Map p<0.005 Axial" data-group="group7" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_p005_mosaic}" alt="Z Map p<0.005 Coronal" data-group="group7" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_p005_mosaic}" alt="Z Map p<0.005 Sagittal" data-group="group7" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
        
                    <div class="threshold-group" data-threshold="p<0.001" style="display: none;">
                        <img src="data:image/png;base64,{axial_zmap_p001_mosaic}" alt="Z Map p<0.001 Axial" data-group="group8" data-view="axial" style="width: 100%; height: auto;" class="active">
                        <img src="data:image/png;base64,{coronal_zmap_p001_mosaic}" alt="Z Map p<0.001 Coronal" data-group="group8" data-view="coronal" style="width: 100%; height: auto;">
                        <img src="data:image/png;base64,{sagittal_zmap_p001_mosaic}" alt="Z Map p<0.001 Sagittal" data-group="group8" data-view="sagittal" style="width: 100%; height: auto;">
                    </div>
                </div>
            </div>
            
            <div class="section highlight">
                <h3>Results</h3>
                <ul>
                    <li><strong>Mean lesion volume (in voxels):</strong> {mean_lesion_volume:.2f}</li>
                    <li><strong>Number of lesion files:</strong> {num_lesions}</li>
                    <li><strong>Z-map value range:</strong> ({zmap_range[0]:.2f}, {zmap_range[1]:.2f})</li>
                </ul>
            </div>
            <div id="section">
                <h2>Interactive Viewer</h2>
                <div>
                    {html_view}
                </div>
            </div>

        """)


        f.write(f"""
                <div class="section">
                    <h3>Download</h3>
                    <p>Download the <a href='zmap.nii.gz'>Z-map NIfTI file</a> for further analysis.</p>
                </div>
            """)

        if fileExists:
            f.write(f"""
                <div class="section">
                <h2>Overview</h2>
                <img src="data:image/png;base64,{overview_image_base64}" alt="Overview Brain Image" style="width: 100%; height: auto;">


                <h2>Cluster Details</h2>

                """)
            for idx, row in top_clusters.iterrows():
                f.write(f"""
                <div class="section highlight" style="margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
                    <img src="data:image/png;base64,{cluster_images_base64[idx]}" style="width: 100%; height: auto; display: block;" alt="Cluster Image">
                    <p style="font-size: 16px; line-height: 1.6; padding-top: 10px;">
                        <strong>Coordinates (X, Y, Z):</strong> ({row['peak_x']}, {row['peak_y']}, {row['peak_z']})<br>
                        <strong>Cluster Mean:</strong> {row['cluster_mean']:.2f}<br>
                        <strong>Volume (mm³):</strong> {row['volume_mm']:.2f}<br>
                        <strong>Desikan-Killiany:</strong><br> {row['desikan_killiany'].replace("_", " ").replace(";", "<br>")}<br>
                        <strong>Harvard-Oxford:</strong><br> {row['harvard_oxford'].replace("_", " ").replace(";", "<br>")}
                    </p>
                </div>
                """)
        else:
            f.write("""
                <div class="section">
                    <p>No significant clusters found</p>
                </div>
            """)
        f.write("""</table></div>

            <div class="footer">
                <p>Generated by SVR-Based Lesion-Symptom Mapping Tool</p>
            </div>

            <!-- Lightbox script -->
            <script>
                function openLightbox(event, element) {
                    event.preventDefault();  // Prevent the default anchor click action
                    var lightbox = document.createElement('div');
                    lightbox.classList.add('lightbox');
                    var img = document.createElement('img');
                    img.src = element.href;
                    lightbox.appendChild(img);
                    document.body.appendChild(lightbox);

                    // Add an event listener to close the lightbox when clicked
                    lightbox.addEventListener('click', function() {
                        lightbox.remove();
                    });
                }
            </script>
            </div>
        </body>
        </html>
        """)

    print(f"Report successfully saved to {output_file}.")


