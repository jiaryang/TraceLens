# we can replay events from the perf reports as well - without the full profile too!
# This is because we essentially require the args and the op name to replay
# excel -> df -> for each row (row -> event -> replayer -> replayer IR -> append to replayer IR list) -> save replayer IR list as json
import pandas as pd
import ast
from TraceLens import EventReplayer
import json
# read sheet from excel

EXCEL_PATH='2025-10-09-modelF-MI325-1n-ddp-deter-logs2.xlsx'
EXCEL_PAGE='ops_unique_args'
COLLUMN_TO_SORT='total_direct_kernel_time_mean'
COLLUMN_TO_COUNT='operation_count'
#OPERATION_LIST=['aten::mm']
OPERATION_LIST=['aten::mm','aten::bmm','aten::addmm']
OUT_PATH='replay_mi325_modelF_1009.json'

df_unique_ops = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_PAGE)

def row_to_evt(row):
    event = {
        'name': row['name'],
        'args': {
            'Input Dims': ast.literal_eval(row['Input Dims']),
            'Input Strides': ast.literal_eval(row['Input Strides']),
            'Input type': ast.literal_eval(row['Input type']),
            'Concrete Inputs': ast.literal_eval(row['Concrete Inputs']),
        }
    }
    return event

repro_data_list = []
processed_count = 0
# lets say we are interested in the following ops

df_ops_interest = df_unique_ops[df_unique_ops['name'].isin(OPERATION_LIST)].copy()

for index, row in df_ops_interest.iterrows():
    event = row_to_evt(row)
    # Initialize EventReplayer similar to above
    replayer = EventReplayer(event, lazy=True, verbose=False)
    # Extract the serializable info
    repro_info = replayer.get_repro_info()
    repro_data_list.append(repro_info)
    processed_count += 1
print(f"Processed {processed_count} events.")
# --- Save the Extracted Data ---
if repro_data_list:
    print(f"\nSaving {len(repro_data_list)} extracted operator infos to '{OUT_PATH}'...")
    with open(OUT_PATH, 'w') as f:
        json.dump(repro_data_list, f, indent=4)
    print("Save complete.")
