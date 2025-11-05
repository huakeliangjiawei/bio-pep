#!/usr/bin/env python3
import os
import json
import io
import requests
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

# 可选导入 RDKit（云端可能不可用）
HAS_RDKIT = True
RDKIT_IMPORT_ERROR = ''
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem import rdDepictor
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetHashedMorganFingerprint
except Exception as e:
    HAS_RDKIT = False
    RDKIT_IMPORT_ERROR = str(e)
    Chem = None
    AllChem = None
    DataStructs = None
    rdDepictor = None
    GetMorganFingerprintAsBitVect = None
    GetHashedMorganFingerprint = None

# 数据路径：优先使用仓库相对 data/，如无则使用 secrets.DATA_URL 指向的 ZIP（自动下载并递归定位）
N_BITS = 1024
RADIUS = 6
TOPK = 10

def _find_dataset_root(base_dir: str):
    meta_path = None
    sdf_dir = None
    for root, dirs, files in os.walk(base_dir):
        if 'metadata.csv' in files and meta_path is None:
            meta_path = os.path.join(root, 'metadata.csv')
        # 找到含有至少一个 .sdf 文件的目录
        if sdf_dir is None:
            for d in dirs:
                cand = os.path.join(root, d)
                try:
                    has_sdf = any(f.lower().endswith('.sdf') for f in os.listdir(cand))
                except Exception:
                    has_sdf = False
                if has_sdf:
                    sdf_dir = cand
                    break
        if meta_path and sdf_dir:
            break
    return meta_path, sdf_dir

def prepare_data_paths():
    # 1) 仓库相对 data/
    repo_data = os.path.join(os.path.dirname(__file__), '..', 'data')
    repo_data = os.path.abspath(repo_data)
    meta1, sdf1 = _find_dataset_root(repo_data)
    if meta1 and sdf1:
        return meta1, sdf1
    # 2) secrets.DATA_URL（ZIP）
    data_url = ''
    try:
        data_url = st.secrets.get('DATA_URL', '').strip()
    except Exception:
        data_url = ''
    if data_url:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'dipeptide_lib')
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, 'dataset.zip')
        if not os.path.exists(zip_path):
            r = requests.get(data_url, timeout=120)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        meta2, sdf2 = _find_dataset_root(cache_dir)
        if meta2 and sdf2:
            return meta2, sdf2
    # 3) fallback：提示缺失
    return None, None

META_CSV, SDF_DIR = prepare_data_paths()

THEME_CSS = """
<style>
:root {
  --warm-amber: #E89B3C;
  --warm-orange: #FF8C42;
  --warm-coral: #FF6F61;
  --ink: #2D3142;
  --muted: #6B7280;
  --paper: #FAF6F1;
}

/* App background */
.stApp {
  background: linear-gradient(180deg, rgba(255,236,215,0.6) 0%, rgba(250,246,241,0.7) 35%, rgba(255,255,255,0.9) 100%),
              radial-gradient(1200px 600px at 10% 0%, rgba(255,161,122,0.18) 0%, rgba(255,161,122,0.0) 70%),
              radial-gradient(900px 500px at 90% 10%, rgba(255,211,158,0.18) 0%, rgba(255,211,158,0.0) 70%);
}

/* Container */
.block-container {
  padding-top: 1rem;
  max-width: 1200px;
}

/* Top hero banner */
.hero {
  width: 100%;
  background: linear-gradient(135deg, var(--warm-amber) 0%, var(--warm-orange) 50%, var(--warm-coral) 100%);
  color: white;
  border-radius: 12px;
  padding: 22px 28px;
  box-shadow: 0 8px 18px rgba(0,0,0,0.12);
  position: relative;
  overflow: hidden;
}
.hero h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0.6px;
}
.hero p {
  margin: 6px 0 0 0;
  color: #fffde7;
}
.hero .molecule {
  position: absolute;
  right: 18px;
  bottom: -8px;
  opacity: 0.22;
}

/* Cards and expanders */
.st-expander {
  background: #fff;
  border: 1px solid rgba(232, 155, 60, 0.25);
  box-shadow: 0 6px 14px rgba(232, 155, 60, 0.15);
  border-radius: 12px !important;
}
.streamlit-expanderHeader {
  font-weight: 600;
  color: var(--ink);
}

/* Buttons */
.stButton>button {
  background: var(--warm-orange);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.5rem 0.9rem;
  font-weight: 600;
  box-shadow: 0 4px 10px rgba(255,140,66,0.3);
}
.stButton>button:hover {
  background: #ff7a22;
}

/* Selects and sliders */
.stSelectbox, .stSlider {
  background: #fffaf4;
  border-radius: 10px;
}

/* Sidebar */
.css-1d391kg, .stSidebar { /* Streamlit internal class may change, best effort */
  background: rgba(255,245,235,0.6);
}

/* Footer */
.footer {
  width: 100%;
  margin-top: 28px;
  padding: 14px 0;
  color: var(--muted);
  border-top: 1px solid rgba(0,0,0,0.08);
  text-align: center;
}
.footer strong { color: var(--ink); }
</style>
"""

HERO_HTML = """
<div class="hero">
  <div style="display:flex; align-items:flex-start; gap:16px;">
    <div style="flex:1 1 auto;">
      <h1>二肽结构相似检索 · Dipeptide Explorer</h1>
      <p>两种查询方式：上传 SDF/SMILES 相似 Top-K；或按 L/D 与氨基酸类型指定检索。指纹：Morgan radius=6, nBits=1024。</p>
    </div>
    <svg class="molecule" width="160" height="160" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#fff" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#fff" stop-opacity="0.4"/>
        </linearGradient>
      </defs>
      <g fill="url(#g)" stroke="white" stroke-opacity="0.6">
        <circle cx="20" cy="50" r="8"/>
        <circle cx="60" cy="40" r="8"/>
        <circle cx="100" cy="60" r="8"/>
        <circle cx="140" cy="50" r="8"/>
        <circle cx="180" cy="70" r="8"/>
        <line x1="20" y1="50" x2="60" y2="40"/>
        <line x1="60" y1="40" x2="100" y2="60"/>
        <line x1="100" y1="60" x2="140" y2="50"/>
        <line x1="140" y1="50" x2="180" y2="70"/>
        <circle cx="40" cy="120" r="8"/>
        <circle cx="80" cy="130" r="8"/>
        <circle cx="120" cy="110" r="8"/>
        <circle cx="160" cy="130" r="8"/>
        <line x1="40" y1="120" x2="80" y2="130"/>
        <line x1="80" y1="130" x2="120" y2="110"/>
        <line x1="120" y1="110" x2="160" y2="130"/>
      </g>
    </svg>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer">
  <span>作者：<strong>Jiawei Liang</strong>; 单位：<strong>Ni Lab in SZU</strong></span>
</div>
"""

@st.cache_resource(show_spinner=False)
def load_resources():
    df = pd.read_csv(META_CSV)
    ids = [str(r['id']) for _, r in df.iterrows()]
    smiles = [str(r['smiles']) for _, r in df.iterrows()]
    id_to_row = {str(r['id']): r for _, r in df.iterrows()}
    # 预生成多种指纹以加速（若 RDKit 不可用则填 None）
    fps = []
    fps_multi = []  # 每个分子一个 dict：{'ECFP6','ECFP4','MACCS166','AtomPair','TopologicalTorsion','LayeredFP'}
    if HAS_RDKIT:
        for smi in smiles:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                fps.append(None)
                fps_multi.append(None)
                continue
            fp_ecfp6 = GetMorganFingerprintAsBitVect(m, radius=6, nBits=N_BITS)
            fp_ecfp4 = GetMorganFingerprintAsBitVect(m, radius=4, nBits=N_BITS)
            try:
                fp_maccs = MACCSkeys.GenMACCSKeys(m)
            except Exception:
                fp_maccs = None
            try:
                fp_atompair = rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=N_BITS)
            except Exception:
                fp_atompair = None
            try:
                fp_tt = rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=N_BITS)
            except Exception:
                fp_tt = None
            try:
                fp_layered = rdMolDescriptors.GetLayeredFingerprint(m)
            except Exception:
                fp_layered = None
            fps.append(fp_ecfp6)
            fps_multi.append({
                'ECFP6': fp_ecfp6,
                'ECFP4': fp_ecfp4,
                'MACCS166': fp_maccs,
                'AtomPair': fp_atompair,
                'TopologicalTorsion': fp_tt,
                'LayeredFP': fp_layered,
            })
    else:
        fps = [None] * len(smiles)
        fps_multi = [None] * len(smiles)
    return df, ids, smiles, fps, id_to_row, fps_multi


def file_uploader_to_mol(uploaded):
    content = uploaded.read()
    # Try SDF first
    try:
        suppl = Chem.ForwardSDMolSupplier(io.BytesIO(content))  # type: ignore
        for m in suppl:
            if m is not None:
                return m
    except Exception:
        pass
    # Try SMILES text
    try:
        txt = content.decode('utf-8').strip()
        m = Chem.MolFromSmiles(txt)
        if m is not None:
            return m
    except Exception:
        pass
    return None


def smiles_to_count_vec(smi: str, nBits: int = N_BITS, radius: int = RADIUS):
    if not HAS_RDKIT:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sv = GetHashedMorganFingerprint(m, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    for k, v in sv.GetNonzeroElements().items():
        arr[k % nBits] = float(v)
    n = np.linalg.norm(arr)
    if n == 0:
        n = 1.0
    return (arr / n).astype(np.float32), m


def get_bit_fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=RADIUS, nBits=N_BITS)


def rerank_with_tanimoto(query_mol, candidate_mols):
    qfp = get_bit_fp(query_mol)
    sims = []
    for cid, mol in candidate_mols:
        if mol is None:
            sims.append((cid, 0.0))
            continue
        cfp = get_bit_fp(mol)
        s = DataStructs.TanimotoSimilarity(qfp, cfp)
        sims.append((cid, float(s)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


def molblock_from_sdf(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()


def render_mol_2d(mol, legend: str = ""):
    if not HAS_RDKIT or mol is None:
        return None
    try:
        # 按需导入绘图后端，避免 Chem/AllChem 可用但 Cairo 缺失时影响整体功能
        from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore
        mc = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mc)
        d2d = rdMolDraw2D.MolDraw2DCairo(280, 210)
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mc, legend=legend)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()
    except Exception:
        return None


def render_mol_3d_py3dmol(sdf_path: str, width=420, height=320):
    import py3Dmol
    molblock = molblock_from_sdf(sdf_path)
    # 移除氢原子以便更清晰展示
    if HAS_RDKIT:
        try:
            suppl = Chem.SDMolSupplier(sdf_path)
            mol = suppl[0] if len(suppl) > 0 else None
            if mol:
                mol_no_h = Chem.RemoveHs(mol, sanitize=True)
                molblock = Chem.MolToMolBlock(mol_no_h)
        except Exception:
            pass
    view = py3Dmol.view(width=width, height=height)
    view.addModel(molblock, 'sdf')
    view.setStyle({'stick': {'colorscheme': 'orangeCarbon'}})
    view.zoomTo()
    return view


def main():
    st.set_page_config(page_title='Dipeptide Explorer', layout='wide', initial_sidebar_state='collapsed')
    # inject theme & hero
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    st.caption(f"RDKit状态：{'可用' if HAS_RDKIT else '不可用'}")
    if not HAS_RDKIT and RDKIT_IMPORT_ERROR:
        st.caption(f"RDKit导入错误：{RDKIT_IMPORT_ERROR}")

    # 数据可用性检查，避免云端缺失文件时报错
    if not (META_CSV and SDF_DIR and os.path.exists(META_CSV) and os.path.isdir(SDF_DIR)):
        st.error('数据未找到：请将 data/metadata.csv 与 data/sdf/ 放入仓库，或在 App Settings -> Secrets 中设置 DATA_URL 指向包含 metadata.csv 与 sdf/ 的 ZIP。')
        st.caption('已自动尝试从 DATA_URL 下载并递归定位文件，但未找到有效结构。请确保 ZIP 根目录包含 metadata.csv 与 sdf/，或其子目录中包含这些内容。')
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return

    df, ids, smiles, fps, id_to_row, fps_multi = load_resources()

    # Optional: load R-side features if present
    r_feat_path = '/data/deepcode/dipeptide_lib/output/peptide_features.csv'
    r_df = None
    if os.path.exists(r_feat_path):
        try:
            r_df = pd.read_csv(r_feat_path)
        except Exception:
            r_df = None

    col1, col2 = st.columns([2, 1])
    with col1:
        if HAS_RDKIT:
            st.subheader('方式一：相似检索')
            query_smiles = st.text_input('输入 SMILES', '')
            uploaded = st.file_uploader('或上传 SDF/SMILES 文件', type=['sdf', 'mol', 'smi', 'txt'])
            topk = st.slider('Top-K 返回数量', 5, 50, TOPK)
            fp_choices = ['ECFP6','ECFP4','MACCS166','AtomPair','TopologicalTorsion','LayeredFP']
            selected_fps = st.multiselect('选择指纹方法（可多选）', fp_choices, default=['ECFP6'])
            metric = st.selectbox('相似度度量', ['Tanimoto','Dice'], index=0)
            agg = st.selectbox('综合方式', ['加权平均','取最大值'], index=0)
            run = st.button('开始检索')
            st.divider()
        else:
            st.info('云端环境未加载 RDKit：已自动隐藏相似检索功能。请确保 requirements 安装成功或固定 Python 版本为 3.10。')
            query_smiles = ''
            uploaded = None
            topk = TOPK
            run = False
        st.subheader('方式二：按 L/D 与氨基酸类型检索')
        ld1 = st.selectbox('第一个氨基酸 L/D', ['L','D'])
        aa1 = st.selectbox('第一个氨基酸类型', sorted(set([x.split('-',1)[1] for x in df['aa1'] if '-' in x])))
        ld2 = st.selectbox('第二个氨基酸 L/D', ['L','D'])
        aa2 = st.selectbox('第二个氨基酸类型', sorted(set([x.split('-',1)[1] for x in df['aa2'] if '-' in x])))
        run_pair = st.button('检索该二肽')
    with col2:
        st.subheader('库统计')
        st.metric('分子数量', f"{len(ids)}")
        st.caption('数据来自扩展非天然 + D 型全集，不去重。')

    query_mol = None
    # Pair query by AA symbols
    if 'run_pair' not in st.session_state:
        st.session_state['run_pair'] = False
    if run_pair:
        st.session_state['run_pair'] = True
    if st.session_state.get('run_pair'):
        n1 = f"{ld1}-{aa1}"
        n2 = f"{ld2}-{aa2}"
        target_seq = f"{n1}-{n2}"
        hits = df[df['seq'] == target_seq]
        if hits.empty:
            st.warning(f'未找到序列 {target_seq} 对应的二肽，请检查氨基酸符号或是否在库内。')
        else:
            st.subheader('按符号检索结果')
            export_rows2 = []
            export_sdf_paths2 = []
            for _, row in hits.iterrows():
                cid = row['id']
                sdf_path = os.path.join(SDF_DIR, f"{cid}.sdf")
                with st.expander(f"{cid}  |  {row['seq']}  |  MW={row['MW']:.1f}  logP={row['logP']:.2f}  TPSA={row['TPSA']:.1f}"):
                    cols = st.columns([1,2])
                    with cols[0]:
                        img = None
                        if HAS_RDKIT:
                            try:
                                img = render_mol_2d(Chem.MolFromSmiles(row['smiles']), legend=cid)
                            except Exception:
                                img = None
                        if img:
                            st.image(img)
                        # 提供无氢版 SDF 下载
                    data_sdf = open(sdf_path,'rb').read()
                    fn = f"{cid}.sdf"
                    if HAS_RDKIT:
                        try:
                            mol = Chem.SDMolSupplier(sdf_path)[0]
                            if mol:
                                mol_no_h = Chem.RemoveHs(mol, sanitize=True)
                                data_sdf = Chem.MolToMolBlock(mol_no_h).encode('utf-8')
                                fn = f"{cid}_noH.sdf"
                        except Exception:
                            pass
                    st.download_button('下载 SDF（无氢）', data=data_sdf, file_name=fn, key=f"dl_sdf_pair_{cid}")
                    with cols[1]:
                        try:
                            view = render_mol_3d_py3dmol(sdf_path)
                            st.components.v1.html(view._make_html(), height=340)
                        except Exception:
                            st.info('3D 视图不可用，显示 2D 图。')
                        if r_df is not None:
                            rr = r_df[r_df['id'] == cid]
                            if not rr.empty:
                                st.markdown('R 侧特征:')
                                rrow = rr.iloc[0]
                                st.write({
                                    'seq_one': rrow.get('seq_one'),
                                    'MW_R': rrow.get('MW_R'),
                                    'hydrophobicity': rrow.get('hydrophobicity'),
                                    'pI': rrow.get('pI'),
                                    'aIndex': rrow.get('aIndex'),
                                    'agree_MW': rrow.get('agree_MW')
                                })
                export_rows2.append({k: row[k] for k in ['id','aa1','aa2','seq','smiles','MW','logP','TPSA']})
                export_sdf_paths2.append(sdf_path)
            # export bundle
            if export_rows2:
                import io, zipfile
                out_csv = io.StringIO()
                pd.DataFrame(export_rows2).to_csv(out_csv, index=False)
                out_csv_bytes = out_csv.getvalue().encode('utf-8')
                st.download_button('下载该序列的 CSV', data=out_csv_bytes, file_name='pair_results.csv', key="dl_pair_csv")
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr('pair_results.csv', out_csv_bytes)
                    for pth in export_sdf_paths2:
                        try:
                            zf.write(pth, arcname=os.path.basename(pth))
                        except Exception:
                            pass
                st.download_button('打包下载该序列 SDF+CSV', data=zbuf.getvalue(), file_name='pair_results_bundle.zip')

    if HAS_RDKIT and run:
        if uploaded is not None:
            import io
            content = uploaded.read()
            try:
                # Try SDF supplier
                suppl = Chem.ForwardSDMolSupplier(io.BytesIO(content))  # type: ignore
                for m in suppl:
                    if m is not None:
                        query_mol = m
                        break
            except Exception:
                pass
            if query_mol is None:
                try:
                    txt = content.decode('utf-8').strip()
                    query_mol = Chem.MolFromSmiles(txt)
                    query_smiles = txt
                except Exception:
                    pass
        if query_mol is None and query_smiles:
            query_mol = Chem.MolFromSmiles(query_smiles)

        if query_mol is None:
            st.error('无法解析输入，请提供有效的 SMILES 或 SDF 文件。')
            st.markdown(FOOTER_HTML, unsafe_allow_html=True)
            return

        # 多指纹 + 多度量 + 综合方式
        def _sim_score(qmol, target_fp_map):
            scores = []
            for name in selected_fps:
                try:
                    if name == 'ECFP6':
                        q = GetMorganFingerprintAsBitVect(qmol, radius=6, nBits=N_BITS)
                        t = target_fp_map.get('ECFP6')
                    elif name == 'ECFP4':
                        q = GetMorganFingerprintAsBitVect(qmol, radius=4, nBits=N_BITS)
                        t = target_fp_map.get('ECFP4')
                    elif name == 'MACCS166':
                        from rdkit.Chem import MACCSkeys
                        q = MACCSkeys.GenMACCSKeys(qmol)
                        t = target_fp_map.get('MACCS166')
                    elif name == 'AtomPair':
                        q = rdMolDescriptors.GetHashedAtomPairFingerprint(qmol, nBits=N_BITS)
                        t = target_fp_map.get('AtomPair')
                    elif name == 'TopologicalTorsion':
                        q = rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(qmol, nBits=N_BITS)
                        t = target_fp_map.get('TopologicalTorsion')
                    elif name == 'LayeredFP':
                        q = rdMolDescriptors.GetLayeredFingerprint(qmol)
                        t = target_fp_map.get('LayeredFP')
                    else:
                        continue
                    if q is None or t is None:
                        continue
                    if metric == 'Tanimoto':
                        s = DataStructs.TanimotoSimilarity(q, t)
                    else:
                        s = DataStructs.DiceSimilarity(q, t)
                    scores.append(float(s))
                except Exception:
                    pass
            if not scores:
                return 0.0
            if agg == '取最大值':
                return max(scores)
            # 默认加权平均（均匀权重）
            return sum(scores) / len(scores)

        sims = []
        for cid, fpmap in zip(ids, fps_multi):
            if fpmap is None:
                sims.append((cid, 0.0))
            else:
                s = _sim_score(query_mol, fpmap)
                sims.append((cid, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        ranked = sims[:topk]

        st.subheader('结果')
        export_rows = []
        export_sdf_paths = []
        for cid, tanimoto in ranked:
            row = id_to_row[cid]
            sdf_path = os.path.join(SDF_DIR, f"{cid}.sdf")
            with st.expander(f"{cid}  |  Tanimoto={tanimoto:.3f}  |  {row['seq']}  |  MW={row['MW']:.1f}  logP={row['logP']:.2f}  TPSA={row['TPSA']:.1f}"):
                cols = st.columns([1,2])
                with cols[0]:
                    img = None
                    if HAS_RDKIT:
                        try:
                            img = render_mol_2d(Chem.MolFromSmiles(row['smiles']), legend=cid)
                        except Exception:
                            img = None
                    if img:
                        st.image(img)
                    # 提供无氢版 SDF 下载
                    data_sdf = open(sdf_path,'rb').read()
                    fn = f"{cid}.sdf"
                    if HAS_RDKIT:
                        try:
                            mol = Chem.SDMolSupplier(sdf_path)[0]
                            if mol:
                                mol_no_h = Chem.RemoveHs(mol, sanitize=True)
                                data_sdf = Chem.MolToMolBlock(mol_no_h).encode('utf-8')
                                fn = f"{cid}_noH.sdf"
                        except Exception:
                            pass
                    st.download_button('下载 SDF（无氢）', data=data_sdf, file_name=fn, key=f"dl_sdf_sim_{cid}")
                with cols[1]:
                    try:
                        view = render_mol_3d_py3dmol(sdf_path)
                        st.components.v1.html(view._make_html(), height=340)
                    except Exception:
                        st.info('3D 视图不可用，显示 2D 图。')
                export_rows.append({k: row[k] for k in ['id','aa1','aa2','seq','smiles','MW','logP','TPSA']})
                export_sdf_paths.append(sdf_path)
        # 打包下载
        if export_rows:
            import io, zipfile
            out_csv = io.StringIO()
            pd.DataFrame(export_rows).to_csv(out_csv, index=False)
            out_csv_bytes = out_csv.getvalue().encode('utf-8')
            st.download_button('下载该结果的 CSV', data=out_csv_bytes, file_name='similar_results.csv')
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('similar_results.csv', out_csv_bytes)
                for pth in export_sdf_paths:
                    try:
                        zf.write(pth, arcname=os.path.basename(pth))
                    except Exception:
                        pass
            st.download_button('打包下载 Top-K SDF+CSV', data=zbuf.getvalue(), file_name='similar_topk_bundle.zip', key="dl_sim_zip")

    # Footer
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
