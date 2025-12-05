import csv
import requests
import os


input_file = r'D:\data_d\result .csv'
output_folder = r"D:\data_d\sdf"
log_file = r'D:\data_d\failed_log.csv'

os.makedirs(output_folder, exist_ok=True)

with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data_list = [(row['ID'], row['SMILES']) for row in reader if row.get('SMILES') and row.get('ID')]

with open(log_file, 'w', newline='', encoding='utf-8') as logf:
    log_writer = csv.writer(logf)
    log_writer.writerow(['原始SMILES', '对应ID', '失败原因'])
    count = 0
    for compound_id, smiles in data_list:
        try:
            res = requests.post(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON",
                data={'smiles': smiles},
                timeout=10
            )
            res.raise_for_status()
        except requests.exceptions.RequestException:
            log_writer.writerow([smiles, compound_id, '网络错误或超时'])
            continue

        data = res.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        if not cids:
            log_writer.writerow([smiles, compound_id, 'SMILES未查到CID'])
            continue
        cid = cids[0]
        if cid == 0:
            log_writer.writerow([smiles, compound_id, 'CID为0'])
            continue

        try:
            sdf_res = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF",
                params={'record_type': '3d'},
                timeout=30
            )
        except requests.exceptions.RequestException:
            log_writer.writerow([smiles, compound_id, '网络错误或超时'])
            continue

        if sdf_res.status_code == 404:
            log_writer.writerow([smiles, compound_id, '无3D结构'])
            continue
        if sdf_res.status_code != 200:
            log_writer.writerow([smiles, compound_id, f'HTTP错误 {sdf_res.status_code}'])
            continue


        sdf_filename = os.path.join(output_folder, f"{compound_id}.sdf")
        with open(sdf_filename, 'w', encoding='utf-8') as sdf_file:
            sdf_file.write(sdf_res.text)
        count+=1
        print(f"成功下载 {count} 个SDF文件")
