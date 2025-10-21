from datetime import datetime
import subprocess

cmd = []
#cmd.append("python lets_try.py --model chess_elo_model_V10.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V10.csv --start 0")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V1.csv --out_prefix results_V1 --save_zero_examples")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V10.csv --out_prefix results_V10 --save_zero_examples")
#cmd.append("python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --max_file 100 --epochs 10 --validation validation_10k_positions_from_130_files.npz --starting_version 10")
#cmd.append("python lets_try.py --model model_versions/chess_elo_model_V20.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V20.csv --start 0")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V20.csv --out_prefix results_V20 --save_zero_examples")
#cmd.append("python csv_to_npz_from_fen_all.py all_positions_jul2014_csv/positions_jul2014.csv all_positions_jul2014_npz/positions_jul2014.npz --json move2idx_all.json --max_games 1048440 --game_split 1500 --max_file 699 --start 131")
#cmd.append("python csv_to_npz_validation.py all_positions_jul2014_csv/positions_jul2014 validation_10k_positions_from_130_files_V2.npz --json move2idx_all.json --selected_indices_out validation_selected_indices_V2.json --example_from_file 77 --game_split 1500 --max_games 1048440 --max_file 130 --start 131")
#cmd.append("python csv_to_npz_validation.py all_positions_jul2014_csv/positions_jul2014 validation_10k_positions_from_130_files_V3.npz --json move2idx_all.json --selected_indices_out validation_selected_indices_V3.json --example_from_file 77 --game_split 1500 --max_games 1048440 --max_file 130 --start 261")
#cmd.append("python csv_to_npz_validation.py all_positions_jul2014_csv/positions_jul2014 validation_10k_positions_from_130_files_V4.npz --json move2idx_all.json --selected_indices_out validation_selected_indices_V4.json --example_from_file 77 --game_split 1500 --max_games 1048440 --max_file 130 --start 391")
#cmd.append("python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --validation validation_10k_positions_from_130_files.npz --validation_indices validation_selected_indices.json --starting_version 20 --start 101 --max_file 30 --epochs 2")
#cmd.append("python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --validation validation_10k_positions_from_130_files_V2.npz --validation_indices validation_selected_indices_V2.json --starting_version 23 --start 101 --max_file 30 --epochs 2")
#cmd.append("python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --validation validation_10k_positions_from_130_files_V3.npz --validation_indices validation_selected_indices_V3.json --starting_version 26 --start 101 --max_file 30 --epochs 2")
#cmd.append("python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --validation validation_10k_positions_from_130_files_V4.npz --validation_indices validation_selected_indices_V4.json --starting_version 29 --start 101 --max_file 30 --epochs 2")
#cmd.append("python lets_try.py --model model_versions/chess_elo_model_V23.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V23.csv --start 0")
#cmd.append("python lets_try.py --model model_versions/chess_elo_model_V26.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V26.csv --start 0")
#cmd.append("python lets_try.py --model model_versions/chess_elo_model_V29.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V29.csv --start 0")
#cmd.append("python lets_try.py --model model_versions/chess_elo_model_V32.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv Validation_Examples_V32.csv --start 0")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V23.csv --out_prefix results_V23 --save_zero_examples")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V26.csv --out_prefix results_V26 --save_zero_examples")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V29.csv --out_prefix results_V29 --save_zero_examples")
#cmd.append("python analyze_intersection.py --csv Validation_Examples_V32.csv --out_prefix results_V32 --save_zero_examples")

for i in range(len(cmd)):
    time = datetime.now().isoformat()
    tot_com = f"Time: {time}, [Batch {i + 1}] Running: {cmd[i]}"
    print(tot_com)
    with open("Command_Log.txt", "a", encoding="utf-8") as f:
        f.write(f"{tot_com}\n\n")
    subprocess.run(f"{cmd[i]}", check=True)