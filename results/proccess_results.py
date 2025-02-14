import os
import pandas as pd
from openpyxl import load_workbook

# Define the root directory
root_dir = './'

# Initialize a dictionary to store results
final_results = {}

# Traverse through each model folder
for model_folder in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_folder)
    if os.path.isdir(model_path) and "pycache" not in model_path:
        
        with_t_simglucose = {'PH_30': [], 'PH_45': [], 'PH_60': [], 'PH_120': []}
        without_t_simglucose = {'PH_30': [], 'PH_45': [], 'PH_60': [], 'PH_120': []}
        
        # Traverse through subfolders inside each model
        for subfolder in os.listdir(model_path):
            subfolder_path = os.path.join(model_path, subfolder)
            if os.path.isdir(subfolder_path):
                
                # Check for 'output.xlsx' in the subfolder
                output_file = os.path.join(subfolder_path, 'output.xlsx')
                if os.path.exists(output_file):
                    workbook = load_workbook(output_file, data_only=True)
                    sheet = workbook.active
                    value_b5 = float(sheet['B5'].value)/0.0001
                    
                    # Identify PH group
                    ph_group = None
                    for ph in ['PH_30', 'PH_45', 'PH_60', 'PH_120']:
                        if ph in subfolder:
                            ph_group = ph
                            break
                    
                    if ph_group:
                        # Append to the respective list
                        if 't_simglucose' in subfolder:
                            with_t_simglucose[ph_group].append(value_b5)
                        else:
                            without_t_simglucose[ph_group].append(value_b5)

        # Store the results for the current model
        final_results[model_folder] = {
            'IIT': with_t_simglucose,
            'NOIIT': without_t_simglucose
        }

# Create a Pandas Excel writer
output_excel = 'aggregated_results.xlsx'
with pd.ExcelWriter(output_excel) as writer:
    for model, data in final_results.items():
        for condition, ph_groups in data.items():
            for ph, values in ph_groups.items():
                df = pd.DataFrame(values, columns=['MSE (mg/dL)2'])
                sheet_name = f'{model.replace("MLP_","")}_{condition}_{ph}'[:31]  # Ensure sheet name length limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f'Results have been saved to {output_excel}')
