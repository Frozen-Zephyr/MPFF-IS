from experiment import *

if __name__ == '__main__':
    file, model, bs, seed,checkcode=parse_args()
    folder=filter_and_save_csv(list(file),checkcode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpnn = MPNNPredictorWithProtein().to(device)
    mpnn.load_state_dict(torch.load(model, weights_only=True))
    mpnn.eval()

    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    file_count = 1
    for csv_file in csv_files:
        print(f'Predicting {file_count} of {len(csv_files)} files...')
        dataset = MolecularDataset.loading_data(csv_file, device=device)
        experiment(model=mpnn,
             dataset=dataset,
             batch_size=bs,
             device=device,
             file=csv_file,
             seed=seed
             )
        print(f"Predicted file saved to {csv_file}！")
        file_count += 1
