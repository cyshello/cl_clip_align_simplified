"""Evaluation"""

import pandas as pd
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


flores_path = "/content/drive/MyDrive/CS371/Final Research/dataset/flores-200"

lang_map = { # trained langugaes
    'ko': 'kor_Hang',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'th': 'tha_Thai',
    'sw': 'swh_Latn',
    'bn': 'ben_Beng',
}

flores_lang_map = { # All languages in flores-200 devtest
    'ace_Arab': 'ace_Arab',
    'ace_Latn': 'ace_Latn',
    'acm_Arab': 'acm_Arab',
    'acq_Arab': 'acq_Arab',
    'aeb_Arab': 'aeb_Arab',
    'afr_Latn': 'afr_Latn',
    'als_Latn': 'als_Latn',
    'amh_Ethi': 'amh_Ethi',
    'apc_Arab_nort3139': 'apc_Arab_nort3139',
    'apc_Arab_sout3123': 'apc_Arab_sout3123',
    'arb_Arab': 'arb_Arab',
    'arb_Latn': 'arb_Latn',
    'arg_Latn': 'arg_Latn',
    'ars_Arab': 'ars_Arab',
    'ary_Arab': 'ary_Arab',
    'arz_Arab': 'arz_Arab',
    'asm_Beng': 'asm_Beng',
    'ast_Latn': 'ast_Latn',
    'awa_Deva': 'awa_Deva',
    'ayr_Latn': 'ayr_Latn',
    'azb_Arab': 'azb_Arab',
    'azj_Latn': 'azj_Latn',
    'bak_Cyrl': 'bak_Cyrl',
    'bam_Latn': 'bam_Latn',
    'ban_Latn': 'ban_Latn',
    'bel_Cyrl': 'bel_Cyrl',
    'bem_Latn': 'bem_Latn',
    'ben_Beng': 'ben_Beng',
    'bho_Deva': 'bho_Deva',
    'bjn_Arab': 'bjn_Arab',
    'bjn_Latn': 'bjn_Latn',
    'bod_Tibt': 'bod_Tibt',
    'bos_Latn': 'bos_Latn',
    'bug_Latn': 'bug_Latn',
    'bul_Cyrl': 'bul_Cyrl',
    'cat_Latn': 'cat_Latn',
    'ceb_Latn': 'ceb_Latn',
    'ces_Latn': 'ces_Latn',
    'chv_Cyrl': 'chv_Cyrl',
    'cjk_Latn': 'cjk_Latn',
    'ckb_Arab': 'ckb_Arab',
    'cmn_Hans': 'cmn_Hans',
    'cmn_Hant': 'cmn_Hant',
    'crh_Latn': 'crh_Latn',
    'cym_Latn': 'cym_Latn',
    'dan_Latn': 'dan_Latn',
    'deu_Latn': 'deu_Latn',
    'dik_Latn': 'dik_Latn',
    'dyu_Latn': 'dyu_Latn',
    'dzo_Tibt': 'dzo_Tibt',
    'ekk_Latn': 'ekk_Latn',
    'ell_Grek': 'ell_Grek',
    'eng_Latn': 'eng_Latn',
    'epo_Latn': 'epo_Latn',
    'eus_Latn': 'eus_Latn',
    'ewe_Latn': 'ewe_Latn',
    'fao_Latn': 'fao_Latn',
    'fij_Latn': 'fij_Latn',
    'fil_Latn': 'fil_Latn',
    'fin_Latn': 'fin_Latn',
    'fon_Latn': 'fon_Latn',
    'fra_Latn': 'fra_Latn',
    'fur_Latn': 'fur_Latn',
    'fuv_Latn': 'fuv_Latn',
    'gaz_Latn': 'gaz_Latn',
    'gla_Latn': 'gla_Latn',
    'gle_Latn': 'gle_Latn',
    'glg_Latn': 'glg_Latn',
    'gug_Latn': 'gug_Latn',
    'guj_Gujr': 'guj_Gujr',
    'hat_Latn': 'hat_Latn',
    'hau_Latn': 'hau_Latn',
    'heb_Hebr': 'heb_Hebr',
    'hin_Deva': 'hin_Deva',
    'hne_Deva': 'hne_Deva',
    'hrv_Latn': 'hrv_Latn',
    'hun_Latn': 'hun_Latn',
    'hye_Armn': 'hye_Armn',
    'ibo_Latn': 'ibo_Latn',
    'ilo_Latn': 'ilo_Latn',
    'ind_Latn': 'ind_Latn',
    'isl_Latn': 'isl_Latn',
    'ita_Latn': 'ita_Latn',
    'jav_Latn': 'jav_Latn',
    'jpn_Jpan': 'jpn_Jpan',
    'kab_Latn': 'kab_Latn',
    'kac_Latn': 'kac_Latn',
    'kam_Latn': 'kam_Latn',
    'kan_Knda': 'kan_Knda',
    'kas_Arab': 'kas_Arab',
    'kas_Deva': 'kas_Deva',
    'kat_Geor': 'kat_Geor',
    'kaz_Cyrl': 'kaz_Cyrl',
    'kbp_Latn': 'kbp_Latn',
    'kea_Latn': 'kea_Latn',
    'khk_Cyrl': 'khk_Cyrl',
    'khm_Khmr': 'khm_Khmr',
    'kik_Latn': 'kik_Latn',
    'kin_Latn': 'kin_Latn',
    'kir_Cyrl': 'kir_Cyrl',
    'kmb_Latn': 'kmb_Latn',
    'kmr_Latn': 'kmr_Latn',
    'knc_Arab': 'knc_Arab',
    'knc_Latn': 'knc_Latn',
    'kor_Hang': 'kor_Hang',
    'ktu_Latn': 'ktu_Latn',
    'lao_Laoo': 'lao_Laoo',
    'lij_Latn': 'lij_Latn',
    'lim_Latn': 'lim_Latn',
    'lin_Latn': 'lin_Latn',
    'lit_Latn': 'lit_Latn',
    'lmo_Latn': 'lmo_Latn',
    'ltg_Latn': 'ltg_Latn',
    'ltz_Latn': 'ltz_Latn',
    'lua_Latn': 'lua_Latn',
    'lug_Latn': 'lug_Latn',
    'luo_Latn': 'luo_Latn',
    'lus_Latn': 'lus_Latn',
    'lvs_Latn': 'lvs_Latn',
    'mag_Deva': 'mag_Deva',
    'mai_Deva': 'mai_Deva',
    'mal_Mlym': 'mal_Mlym',
    'mar_Deva': 'mar_Deva',
    'min_Arab': 'min_Arab',
    'min_Latn': 'min_Latn',
    'mkd_Cyrl': 'mkd_Cyrl',
    'mlt_Latn': 'mlt_Latn',
    'mni_Beng': 'mni_Beng',
    'mni_Mtei': 'mni_Mtei',
    'mos_Latn': 'mos_Latn',
    'mri_Latn': 'mri_Latn',
    'mya_Mymr': 'mya_Mymr',
    'myv_Cyrl': 'myv_Cyrl',
    'nld_Latn': 'nld_Latn',
    'nno_Latn': 'nno_Latn',
    'nob_Latn': 'nob_Latn',
    'npi_Deva': 'npi_Deva',
    'nqo_Nkoo': 'nqo_Nkoo',
    'nso_Latn': 'nso_Latn',
    'nus_Latn': 'nus_Latn',
    'nya_Latn': 'nya_Latn',
    'oci_Latn': 'oci_Latn',
    'oci_Latn_aran1260': 'oci_Latn_aran1260',
    'ory_Orya': 'ory_Orya',
    'pag_Latn': 'pag_Latn',
    'pan_Guru': 'pan_Guru',
    'pap_Latn': 'pap_Latn',
    'pbt_Arab': 'pbt_Arab',
    'pes_Arab': 'pes_Arab',
    'plt_Latn': 'plt_Latn',
    'pol_Latn': 'pol_Latn',
    'por_Latn': 'por_Latn',
    'prs_Arab': 'prs_Arab',
    'quy_Latn': 'quy_Latn',
    'ron_Latn': 'ron_Latn',
    'run_Latn': 'run_Latn',
    'rus_Cyrl': 'rus_Cyrl',
    'sag_Latn': 'sag_Latn',
    'san_Deva': 'san_Deva',
    'sat_Olck': 'sat_Olck',
    'scn_Latn': 'scn_Latn',
    'shn_Mymr': 'shn_Mymr',
    'sin_Sinh': 'sin_Sinh',
    'slk_Latn': 'slk_Latn',
    'slv_Latn': 'slv_Latn',
    'smo_Latn': 'smo_Latn',
    'sna_Latn': 'sna_Latn',
    'snd_Arab': 'snd_Arab',
    'som_Latn': 'som_Latn',
    'sot_Latn': 'sot_Latn',
    'spa_Latn': 'spa_Latn',
    'srd_Latn': 'srd_Latn',
    'srp_Cyrl': 'srp_Cyrl',
    'ssw_Latn': 'ssw_Latn',
    'sun_Latn': 'sun_Latn',
    'swe_Latn': 'swe_Latn',
    'swh_Latn': 'swh_Latn',
    'szl_Latn': 'szl_Latn',
    'tam_Taml': 'tam_Taml',
    'taq_Latn': 'taq_Latn',
    'taq_Tfng': 'taq_Tfng',
    'tat_Cyrl': 'tat_Cyrl',
    'tel_Telu': 'tel_Telu',
    'tgk_Cyrl': 'tgk_Cyrl',
    'tha_Thai': 'tha_Thai',
    'tir_Ethi': 'tir_Ethi',
    'tpi_Latn': 'tpi_Latn',
    'tsn_Latn': 'tsn_Latn',
    'tso_Latn': 'tso_Latn',
    'tuk_Latn': 'tuk_Latn',
    'tum_Latn': 'tum_Latn',
    'tur_Latn': 'tur_Latn',
    'twi_Latn_akua1239': 'twi_Latn_akua1239',
    'twi_Latn_asan1239': 'twi_Latn_asan1239',
    'tyv_Cyrl': 'tyv_Cyrl',
    'uig_Arab': 'uig_Arab',
    'ukr_Cyrl': 'ukr_Cyrl',
    'umb_Latn': 'umb_Latn',
    'urd_Arab': 'urd_Arab',
    'uzn_Latn': 'uzn_Latn',
    'vec_Latn': 'vec_Latn',
    'vie_Latn': 'vie_Latn',
    'vmw_Latn': 'vmw_Latn',
    'war_Latn': 'war_Latn',
    'wol_Latn': 'wol_Latn',
    'xho_Latn': 'xho_Latn',
    'ydd_Hebr': 'ydd_Hebr',
    'yor_Latn': 'yor_Latn',
    'yue_Hant': 'yue_Hant',
    'zgh_Tfng': 'zgh_Tfng',
    'zsm_Latn': 'zsm_Latn',
    'zul_Latn': 'zul_Latn',
}

def load_flores_sentences(parquet_dir, lang_code):
    file_path = os.path.join(parquet_dir, f"{lang_code}.parquet")
    df = pd.read_parquet(file_path)
    #print(df.columns)
    if 'text' in df.columns:
        sentences = df['text'].astype(str).tolist()
    else:
        sentences = df.iloc[:, 0].astype(str).tolist()  
    return sentences

@torch.no_grad()
def encode_sentences(model, encoder, sentences, batch_size=64):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        embs = encoder(batch)
        embs = F.normalize(embs, dim=1)
        embeddings.append(embs.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def bitext_retrieval_accuracy(src_embeds, tgt_embeds):
    sims = cosine_similarity(src_embeds, tgt_embeds)
    preds = np.argmax(sims, axis=1)
    correct = np.sum(preds == np.arange(len(src_embeds)))
    return correct / len(src_embeds)

def tsne_visualization(src_embeds, tgt_embeds, src_lang, tgt_lang, n_samples=1000):
    src_embeds = src_embeds[:n_samples]
    tgt_embeds = tgt_embeds[:n_samples]

    combined = np.vstack([src_embeds, tgt_embeds])
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(combined)

    src_2d = reduced[:n_samples]
    tgt_2d = reduced[n_samples:]

    plt.figure(figsize=(10,10))
    plt.scatter(src_2d[:,0], src_2d[:,1], c='red', label=src_lang, alpha=0.6)
    plt.scatter(tgt_2d[:,0], tgt_2d[:,1], c='blue', label=tgt_lang, alpha=0.6)

    for i in range(n_samples):
        plt.plot([src_2d[i,0], tgt_2d[i,0]], [src_2d[i,1], tgt_2d[i,1]], c='gray', alpha=0.2)

    plt.legend()
    plt.title(f"t-SNE of {src_lang} and {tgt_lang} embeddings")
    plt.show()

def evaluate_flores_plus(model, encoder, flores_dir, lang_map, ref_lang='eng_Latn', batch_size=64, visual=True):
    results = {}
    ref_sentences = load_flores_sentences(flores_dir, ref_lang)
    #print(ref_sentences)
    ref_embeds = encode_sentences(model,encoder, ref_sentences, batch_size)

    for short_code, lang_code in lang_map.items():
        print(f"Evaluating {lang_code} vs {ref_lang}")
        try:
            src_sentences = load_flores_sentences(flores_dir, lang_code)
            #print(src_sentences)

        except FileNotFoundError:
            print(f"Parquet file for {lang_code} not found. Skipping...")
            continue

        src_embeds = encode_sentences(model, encoder, src_sentences, batch_size)
        acc = bitext_retrieval_accuracy(src_embeds, ref_embeds)
        print(f"Bitext retrieval accuracy ({lang_code} → {ref_lang}): {acc*100:.2f}%")
        results[lang_code] = acc

    if visual:
        # 대표 언어로 첫 언어 선택하여 t-SNE 시각화 (첫 언어가 parquet 없으면 건너뜀)
        for lang_code in lang_map.values():
            if lang_code in results:
                src_sentences = load_flores_sentences(flores_dir, lang_code)
                src_embeds = encode_sentences(model,encoder, src_sentences, batch_size)
                tsne_visualization(src_embeds, ref_embeds, lang_code, ref_lang)
                #break

    return results

# evalutate baseline encoder : XLM-R
def xlm_r_evaluate():
    import torch
    import torch.nn.functional as F
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) XLM-RoBERTa 토크나이저와 모델 불러오기
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    xlmr_model = XLMRobertaModel.from_pretrained("xlm-roberta-large").to(device)
    xlmr_model.eval()

    # 2) encoder 함수 정의: 문장 리스트 → 문장 임베딩 반환
    @torch.no_grad()
    def xlmr_encoder(sentences, batch_size=64):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = xlmr_model(**encoded)
            last_hidden = outputs.last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(pooled.cpu())  # tensor 반환, normalize 안함
        return torch.cat(embeddings, dim=0)

    results = evaluate_flores_plus(None, xlmr_encoder, flores_path, flores_lang_map, ref_lang='eng_Latn',visual=False)
    return results