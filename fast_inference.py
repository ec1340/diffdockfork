import os
import copy
import torch
import numpy as np
import yaml
from types import SimpleNamespace

from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset
from utils.sampling import randomize_position, sampling
from utils.utils import get_model


def run_fast_inference(ligand_smiles: str, protein_sequence: str, config_path: str = 'default_inference_args.yaml'):
    """
    Minimal inference function for a single protein-ligand pair, requiring only ligand SMILES and protein sequence.
    All other parameters are loaded from a YAML config file.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set up all parameters from config
    model_dir = config['model_dir']
    ckpt = config['ckpt']
    out_dir = config.get('out_dir', 'results/user_inference')
    complex_name = config.get('complex_name', 'complex_0')
    samples_per_complex = config.get('samples_per_complex', 10)
    batch_size = config.get('batch_size', 10)
    inference_steps = config.get('inference_steps', 20)
    actual_steps = config.get('actual_steps', None)
    no_final_step_noise = config.get('no_final_step_noise', True)
    initial_noise_std_proportion = config.get('initial_noise_std_proportion', -1.0)
    choose_residue = config.get('choose_residue', False)
    save_visualisation = config.get('save_visualisation', False)
    confidence_model_dir = config.get('confidence_model_dir', None)
    confidence_ckpt = config.get('confidence_ckpt', 'best_model.pt')
    temp_sampling_tr = config.get('temp_sampling_tr', 1.0)
    temp_psi_tr = config.get('temp_psi_tr', 0.0)
    temp_sigma_data_tr = config.get('temp_sigma_data_tr', 0.5)
    temp_sampling_rot = config.get('temp_sampling_rot', 1.0)
    temp_psi_rot = config.get('temp_psi_rot', 0.0)
    temp_sigma_data_rot = config.get('temp_sigma_data_rot', 0.5)
    temp_sampling_tor = config.get('temp_sampling_tor', 1.0)
    temp_psi_tor = config.get('temp_psi_tor', 0.0)
    temp_sigma_data_tor = config.get('temp_sigma_data_tor', 0.5)
    old_score_model = config.get('old_score_model', False)
    old_confidence_model = config.get('old_confidence_model', True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'model_parameters.yml')) as f:
        score_model_args = SimpleNamespace(**yaml.full_load(f))
    if confidence_model_dir is not None:
        with open(os.path.join(confidence_model_dir, 'model_parameters.yml')) as f:
            confidence_args = SimpleNamespace(**yaml.full_load(f))
    else:
        confidence_args = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset for a single complex
    complex_name_list = [complex_name]
    protein_path_list = [None]  # Not used
    protein_sequence_list = [protein_sequence]
    ligand_description_list = [ligand_smiles]
    for name in complex_name_list:
        write_dir = os.path.join(out_dir, name)
        os.makedirs(write_dir, exist_ok=True)

    test_dataset = InferenceDataset(
        out_dir=out_dir,
        complex_names=complex_name_list,
        protein_files=protein_path_list,
        ligand_descriptions=ligand_description_list,
        protein_sequences=protein_sequence_list,
        lm_embeddings=True,
        receptor_radius=score_model_args.receptor_radius,
        remove_hs=score_model_args.remove_hs,
        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
        all_atoms=score_model_args.all_atoms,
        atom_radius=score_model_args.atom_radius,
        atom_max_neighbors=score_model_args.atom_max_neighbors,
        knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    t_to_sigma = lambda t: t_to_sigma_compl(t, args=score_model_args)
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=old_score_model)
    state_dict = torch.load(os.path.join(model_dir, ckpt), map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if confidence_model_dir is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True, old=old_confidence_model)
        state_dict = torch.load(os.path.join(confidence_model_dir, confidence_ckpt), map_location='cpu')
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None

    tr_schedule = get_t_schedule(inference_steps=inference_steps, sigma_schedule='expbeta')
    N = samples_per_complex

    for idx, orig_complex_graph in enumerate(test_loader):
        if not orig_complex_graph.success[0]:
            continue
        if confidence_model is not None and confidence_args is not None and not confidence_args.use_original_model_cache:
            confidence_complex_graph = InferenceDataset(
                out_dir=out_dir,
                complex_names=complex_name_list,
                protein_files=protein_path_list,
                ligand_descriptions=ligand_description_list,
                protein_sequences=protein_sequence_list,
                lm_embeddings=True,
                receptor_radius=confidence_args.receptor_radius,
                remove_hs=confidence_args.remove_hs,
                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                all_atoms=confidence_args.all_atoms,
                atom_radius=confidence_args.atom_radius,
                atom_max_neighbors=confidence_args.atom_max_neighbors,
                precomputed_lm_embeddings=test_dataset.lm_embeddings,
                knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
            )[idx]
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(
            data_list,
            score_model_args.no_torsion,
            False,
            score_model_args.tr_sigma_max,
            initial_noise_std_proportion=initial_noise_std_proportion,
            choose_residue=choose_residue
        )
        lig = orig_complex_graph.mol[0]
        visualization_list = None  # Minimal version skips visualisation
        data_list, confidence = sampling(
            data_list=data_list,
            model=model,
            inference_steps=actual_steps if actual_steps is not None else inference_steps,
            tr_schedule=tr_schedule,
            rot_schedule=tr_schedule,
            tor_schedule=tr_schedule,
            device=device,
            t_to_sigma=t_to_sigma,
            model_args=score_model_args,
            visualization_list=visualization_list,
            confidence_model=confidence_model,
            confidence_data_list=confidence_data_list,
            confidence_model_args=confidence_args,
            batch_size=batch_size,
            no_final_step_noise=no_final_step_noise,
            temp_sampling=[temp_sampling_tr, temp_sampling_rot, temp_sampling_tor],
            temp_psi=[temp_psi_tr, temp_psi_rot, temp_psi_tor],
            temp_sigma_data=[temp_sigma_data_tr, temp_sigma_data_rot, temp_sigma_data_tor]
        )
        ligand_pos = np.asarray([
            complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
            for complex_graph in data_list
        ])
        if confidence is not None and confidence_args and isinstance(confidence_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]
        write_dir = os.path.join(out_dir, complex_name_list[idx])
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs:
                mol_pred = RemoveAllHs(mol_pred)
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
        return ligand_pos, confidence  # Return results for further use in scripts
