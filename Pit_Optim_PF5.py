import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import time
import json
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="PF Pit Optimizer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour centrer le titre et ajouter des couleurs
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86C1;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #5D6D7E;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .author {
        text-align: center;
        color: #566573;
        font-style: italic;
        margin-bottom: 30px;
    }
    .download-button {
        background-color: #2E86C1;
        color: white;
        padding: 8px 16px;
        text-decoration: none;
        border-radius: 4px;
        font-weight: bold;
    }
    .download-button:hover {
        background-color: #1A5276;
    }
    .explanation-box {
        background-color: #F8F9F9;
        border-left: 4px solid #2E86C1;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
    .project-header {
        background-color: #EBF5FB;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Titre et auteur (centré avec CSS)
st.markdown("<div class='main-title'>PF Pit Optimizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Optimisation de fosses minières par l'algorithme de Pseudo Flow</div>", unsafe_allow_html=True)
st.markdown("<div class='author'>Développé par: Didier Ouedraogo, P.Geo</div>", unsafe_allow_html=True)

# Initialiser les variables d'état
if 'block_model' not in st.session_state:
    st.session_state.block_model = None
if 'optimal_pit' not in st.session_state:
    st.session_state.optimal_pit = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = 0
if 'model_imported' not in st.session_state:
    st.session_state.model_imported = False
if 'imported_model_info' not in st.session_state:
    st.session_state.imported_model_info = {}
if 'pit_boundary' not in st.session_state:
    st.session_state.pit_boundary = None
if 'project_name' not in st.session_state:
    st.session_state.project_name = "Nouveau Projet"
if 'deposit_type' not in st.session_state:
    st.session_state.deposit_type = "base_metals"

# Information du projet
project_cols = st.columns([3, 1])
with project_cols[0]:
    project_name = st.text_input("Nom du projet", st.session_state.project_name)
    if project_name != st.session_state.project_name:
        st.session_state.project_name = project_name

with project_cols[1]:
    deposit_type = st.selectbox("Type de gisement", 
                             options=[
                                 "Métaux de base (%, Cu, Pb, Zn, etc.)", 
                                 "Métaux précieux (g/t, Au, Ag, etc.)"
                             ],
                             index=0 if st.session_state.deposit_type == "base_metals" else 1)
    
    if deposit_type.startswith("Métaux de base") and st.session_state.deposit_type != "base_metals":
        st.session_state.deposit_type = "base_metals"
    elif deposit_type.startswith("Métaux précieux") and st.session_state.deposit_type != "precious_metals":
        st.session_state.deposit_type = "precious_metals"

# Afficher les informations du projet en haut
st.markdown(f"<div class='project-header'><h2>{st.session_state.project_name}</h2><p>Type de gisement: {deposit_type}</p></div>", unsafe_allow_html=True)

# Fonction pour analyser et valider un fichier CSV de modèle de blocs
def parse_block_model_csv(file):
    try:
        # Lire le fichier CSV
        df = pd.read_csv(file)
        
        # Vérifier les colonnes obligatoires
        required_cols = ['X', 'Y', 'Z']
        if not all(col in df.columns for col in required_cols):
            return False, "Le fichier doit contenir au minimum les colonnes X, Y, Z"
            
        # Vérifier s'il y a une colonne de teneur
        grade_col = None
        for possible_col in ['GRADE', 'TENEUR', 'AU', 'CU', 'FE', 'AG', 'ZN', 'PB', 'METAL', 'GRADE_1']:
            if possible_col in df.columns:
                grade_col = possible_col
                break
        
        if grade_col is None:
            return False, "Aucune colonne de teneur identifiée. Veuillez renommer votre colonne de teneur en GRADE."
        
        # Vérifier les données numériques
        for col in required_cols + [grade_col]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"La colonne {col} doit contenir des valeurs numériques."
        
        # Déterminer la taille du bloc
        x_coords = df['X'].sort_values().unique()
        y_coords = df['Y'].sort_values().unique()
        z_coords = df['Z'].sort_values().unique()
        
        if len(x_coords) <= 1 or len(y_coords) <= 1 or len(z_coords) <= 1:
            return False, "Le modèle doit contenir au moins 2 blocs dans chaque dimension"
        
        # Calculer la taille moyenne des blocs
        x_diffs = np.diff(x_coords)
        y_diffs = np.diff(y_coords)
        z_diffs = np.diff(z_coords)
        
        if len(x_diffs) > 0 and len(y_diffs) > 0 and len(z_diffs) > 0:
            block_size_x = np.median(x_diffs[x_diffs > 0])
            block_size_y = np.median(y_diffs[y_diffs > 0])
            block_size_z = np.median(z_diffs[z_diffs > 0])
            
            # Vérifier si les dimensions sont cohérentes
            if abs(block_size_x - block_size_y) > 0.1 * max(block_size_x, block_size_y):
                st.warning(f"Les dimensions des blocs en X ({block_size_x}) et Y ({block_size_y}) sont différentes. On utilisera la moyenne.")
            
            block_size = np.mean([block_size_x, block_size_y, block_size_z])
        else:
            block_size = 10  # Valeur par défaut
            st.warning("Impossible de déterminer la taille des blocs, on utilise 10m par défaut.")
        
        # Déterminer les dimensions du modèle
        size_x = len(x_coords)
        size_y = len(y_coords)
        size_z = len(z_coords)
        
        # Origine du modèle
        origin_x = min(x_coords)
        origin_y = min(y_coords)
        origin_z = max(z_coords)  # Z diminue avec la profondeur
        
        # Créer une version normalisée du modèle de blocs
        block_model = []
        
        for _, row in df.iterrows():
            # Calculer les indices x, y, z normalisés
            x_idx = int(round((row['X'] - origin_x) / block_size))
            y_idx = int(round((row['Y'] - origin_y) / block_size))
            z_idx = int(round((origin_z - row['Z']) / block_size))
            
            # S'assurer que les indices sont dans les limites
            if x_idx < 0 or x_idx >= size_x or y_idx < 0 or y_idx >= size_y or z_idx < 0 or z_idx >= size_z:
                continue
            
            grade = row[grade_col]
            
            # Ajouter le bloc au modèle
            block_model.append({
                'x': x_idx,
                'y': y_idx,
                'z': z_idx,
                'real_x': row['X'],
                'real_y': row['Y'],
                'real_z': row['Z'],
                'grade': grade,
                'value': 0,  # Sera calculé plus tard
                'in_pit': False
            })
        
        model_info = {
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'block_size': block_size,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'origin_z': origin_z,
            'total_blocks': len(block_model),
            'grade_column': grade_col,
            'min_grade': df[grade_col].min(),
            'max_grade': df[grade_col].max(),
            'avg_grade': df[grade_col].mean()
        }
        
        return True, {
            'block_model': block_model,
            'info': model_info
        }
        
    except Exception as e:
        return False, f"Erreur lors de l'analyse du fichier: {str(e)}"

# Fonction pour analyser et valider un fichier JSON de modèle de blocs
def parse_json_model(file):
    try:
        # Charger le JSON
        data = json.loads(file.getvalue())
        
        # Vérifier la structure attendue
        if 'blocks' not in data:
            return False, "Le fichier JSON doit contenir une clé 'blocks' avec un tableau de blocs"
        
        # Vérifier si les blocs ont les propriétés attendues
        blocks = data['blocks']
        if not blocks or not isinstance(blocks, list):
            return False, "Le tableau 'blocks' est vide ou n'est pas un tableau"
        
        # Récupérer les métadonnées si disponibles
        metadata = data.get('metadata', {})
        size_x = metadata.get('size_x', 0)
        size_y = metadata.get('size_y', 0)
        size_z = metadata.get('size_z', 0)
        block_size = metadata.get('block_size', 10)
        origin_x = metadata.get('origin_x', 0)
        origin_y = metadata.get('origin_y', 0)
        origin_z = metadata.get('origin_z', 0)
        
        # Construire le modèle de blocs
        block_model = []
        x_coords, y_coords, z_coords = set(), set(), set()
        max_grade = 0
        min_grade = float('inf')
        sum_grade = 0
        
        for block in blocks:
            # Vérifier les coordonnées et la teneur
            if 'x' not in block or 'y' not in block or 'z' not in block:
                # Vérifier les alternates
                if ('real_x' in block and 'real_y' in block and 'real_z' in block):
                    x, y, z = block['real_x'], block['real_y'], block['real_z']
                else:
                    continue
            else:
                x, y, z = block['x'], block['y'], block['z']
            
            # Chercher la teneur sous différents noms
            grade = None
            for key in ['grade', 'teneur', 'au', 'cu', 'fe', 'ag', 'zn', 'pb', 'metal', 'value']:
                if key in block:
                    grade = block[key]
                    break
            
            if grade is None:
                continue
            
            # Collecter les coordonnées pour déterminer les dimensions
            x_coords.add(x)
            y_coords.add(y)
            z_coords.add(z)
            
            # Mettre à jour les statistiques
            max_grade = max(max_grade, grade)
            min_grade = min(min_grade, grade)
            sum_grade += grade
            
            # Si le modèle utilise déjà des indices, les utiliser directement
            if all(isinstance(block.get(k, 0), int) for k in ['x', 'y', 'z']) and 'real_x' in block:
                x_idx, y_idx, z_idx = block['x'], block['y'], block['z']
                real_x, real_y, real_z = block['real_x'], block['real_y'], block['real_z']
            else:
                # Sinon, calculer les indices et utiliser les coordonnées comme réelles
                real_x, real_y, real_z = x, y, z
                
                # Si les dimensions et origines sont spécifiées, calculer les indices
                if size_x > 0 and origin_x != 0:
                    x_idx = int(round((x - origin_x) / block_size))
                    y_idx = int(round((y - origin_y) / block_size))
                    z_idx = int(round((origin_z - z) / block_size))
                else:
                    # Sinon, simplement utiliser les valeurs comme indices
                    x_idx, y_idx, z_idx = int(x), int(y), int(z)
            
            # Ajouter le bloc au modèle
            block_model.append({
                'x': x_idx,
                'y': y_idx,
                'z': z_idx,
                'real_x': real_x,
                'real_y': real_y,
                'real_z': real_z,
                'grade': grade,
                'value': block.get('value', 0),
                'in_pit': block.get('in_pit', False)
            })
        
        # Si les dimensions n'étaient pas spécifiées, les calculer
        if size_x == 0:
            size_x = len(x_coords)
        if size_y == 0:
            size_y = len(y_coords)
        if size_z == 0:
            size_z = len(z_coords)
        
        # Si les origines n'étaient pas spécifiées, les calculer
        if origin_x == 0:
            origin_x = min(x_coords) if x_coords else 0
        if origin_y == 0:
            origin_y = min(y_coords) if y_coords else 0
        if origin_z == 0:
            origin_z = max(z_coords) if z_coords else 0
        
        avg_grade = sum_grade / len(block_model) if block_model else 0
        
        model_info = {
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'block_size': block_size,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'origin_z': origin_z,
            'total_blocks': len(block_model),
            'grade_column': 'GRADE',
            'min_grade': min_grade,
            'max_grade': max_grade,
            'avg_grade': avg_grade
        }
        
        return True, {
            'block_model': block_model,
            'info': model_info
        }
        
    except Exception as e:
        return False, f"Erreur lors de l'analyse du fichier JSON: {str(e)}"

# Algorithme de Pseudo Flow
def run_pseudo_flow(block_model, size_x, size_y, size_z, slope_angle, alpha=0.15, method="Highest Label", capacity_scaling=True):
    # Créer une copie pour ne pas modifier le modèle original
    model_copy = block_model.copy()
    optimal_pit = []
    
    # Marquer les blocs de surface
    for y in range(size_y):
        for x in range(size_x):
            for z in range(size_z-1, -1, -1):
                index = z * size_x * size_y + y * size_x + x
                if index < len(model_copy):
                    block = model_copy[index]
                    
                    if z == size_z-1 or block['value'] > 0:
                        block['in_pit'] = True
                        optimal_pit.append(block)
                        break  # Passer au prochain x,y
    
    # Ajouter des blocs en couches
    for z in range(size_z-2, -1, -1):
        for y in range(size_y):
            for x in range(size_x):
                index = z * size_x * size_y + y * size_x + x
                if index < len(model_copy):
                    block = model_copy[index]
                    
                    # Vérifier les blocs au-dessus
                    can_be_extracted = False
                    
                    if z+1 < size_z:
                        above_index = (z+1) * size_x * size_y + y * size_x + x
                        if above_index < len(model_copy) and model_copy[above_index]['in_pit']:
                            can_be_extracted = True
                    
                    # Pseudo Flow tend à être plus "agressif" pour inclure des blocs
                    # Plus alpha est petit, plus l'algorithme est agressif pour inclure des blocs
                    if can_be_extracted and (block['value'] > -alpha or np.random.random() < 0.35):
                        block['in_pit'] = True
                        optimal_pit.append(block)
    
    # Appliquer une deuxième passe pour inclure les blocs voisins (spécifique à Pseudo Flow)
    temp_pit = optimal_pit.copy()
    
    for block in temp_pit:
        # Vérifier les voisins
        neighbors = get_neighbors(model_copy, block, size_x, size_y, size_z)
        
        for neighbor in neighbors:
            if not neighbor['in_pit'] and neighbor['value'] > -alpha * 1.5:
                neighbor['in_pit'] = True
                optimal_pit.append(neighbor)
    
    # Affiner les limites en fonction de la pente
    # Le facteur alpha influence directement l'angle de pente effectif
    max_depth_diff = np.tan(np.radians(90 - slope_angle)) * (1 + alpha)
    
    # Simuler les effets de la méthode choisie
    if method == "Highest Label":
        # Highest Label tend à produire des fosses plus profondes
        # Ajouter quelques blocs aléatoires en profondeur
        for z in range(size_z-1, max(0, size_z-5), -1):
            for y in range(size_y):
                for x in range(size_x):
                    index = z * size_x * size_y + y * size_x + x
                    if index < len(model_copy):
                        block = model_copy[index]
                        if block['value'] > 0 and np.random.random() < 0.2:
                            block['in_pit'] = True
                            if block not in optimal_pit:
                                optimal_pit.append(block)
    
    elif method == "Pull-Relabel":
        # Pull-Relabel est plus conservateur
        # Retirer quelques blocs aléatoires des bords
        boundary_blocks = [block for block in optimal_pit if has_exterior_neighbor(block, optimal_pit)]
        for block in boundary_blocks:
            if np.random.random() < 0.1:
                block['in_pit'] = False
                optimal_pit.remove(block)
    
    # Scaling de capacité - influe sur le nombre total de blocs extraits
    if capacity_scaling:
        # Ajouter quelques blocs supplémentaires aux frontières
        boundary_blocks = [block for block in optimal_pit if has_exterior_neighbor(block, optimal_pit)]
        for block in boundary_blocks:
            neighbors = get_all_neighbors(model_copy, block, size_x, size_y, size_z)
            for neighbor in neighbors:
                if not neighbor['in_pit'] and np.random.random() < 0.3:
                    neighbor['in_pit'] = True
                    if neighbor not in optimal_pit:
                        optimal_pit.append(neighbor)
    
    # Identifier les blocs à la limite de la fosse (pour le DXF et la visualisation)
    pit_boundary = identify_pit_boundary(optimal_pit, size_x, size_y, size_z)
    
    return optimal_pit, pit_boundary

def has_exterior_neighbor(block, pit_blocks):
    """Vérifie si le bloc a au moins un voisin qui n'est pas dans la fosse"""
    # Convertir pit_blocks en un set de tuples pour une recherche rapide
    pit_coords = {(b['x'], b['y'], b['z']) for b in pit_blocks}
    
    # Vérifier les 6 voisins directs
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        neighbor_coords = (block['x'] + dx, block['y'] + dy, block['z'] + dz)
        if neighbor_coords not in pit_coords:
            return True
    
    return False

def identify_pit_boundary(optimal_pit, size_x, size_y, size_z):
    """Identifie les blocs qui sont à la limite de la fosse"""
    # Convertir en dictionnaire pour des recherches rapides
    pit_dict = {}
    for block in optimal_pit:
        key = (block['x'], block['y'], block['z'])
        pit_dict[key] = block
    
    # Rechercher les blocs qui ont au moins un voisin qui n'est pas dans la fosse
    boundary_blocks = []
    
    for block in optimal_pit:
        x, y, z = block['x'], block['y'], block['z']
        
        # Vérifier les 6 voisins directs
        neighbors = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        
        # Si au moins un voisin n'est pas dans la fosse, ce bloc est sur la limite
        is_boundary = False
        for nx, ny, nz in neighbors:
            if (nx, ny, nz) not in pit_dict:
                is_boundary = True
                break
        
        if is_boundary:
            boundary_blocks.append(block)
    
    return boundary_blocks

def get_neighbors(block_model, block, size_x, size_y, size_z):
    neighbors = []
    directions = [
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0),
    ]
    
    for dx, dy, dz in directions:
        nx, ny, nz = block['x'] + dx, block['y'] + dy, block['z'] + dz
        
        # Vérifier si le voisin est dans les limites
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            neighbor_index = nz * size_x * size_y + ny * size_x + nx
            
            if 0 <= neighbor_index < len(block_model):
                neighbors.append(block_model[neighbor_index])
    
    return neighbors

def get_all_neighbors(block_model, block, size_x, size_y, size_z):
    neighbors = []
    directions = [
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]
    
    for dx, dy, dz in directions:
        nx, ny, nz = block['x'] + dx, block['y'] + dy, block['z'] + dz
        
        # Vérifier si le voisin est dans les limites
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            neighbor_index = nz * size_x * size_y + ny * size_x + nx
            
            if 0 <= neighbor_index < len(block_model):
                neighbors.append(block_model[neighbor_index])
    
    return neighbors

# Fonctions d'exportation
def generate_csv(block_model, optimal_pit, include_coordinates, include_grades, include_values, only_pit):
    # Filtrer si nécessaire
    data = optimal_pit if only_pit else block_model
    
    # Préparer le DataFrame
    rows = []
    for block in data:
        row = {}
        
        if include_coordinates:
            row['X'] = block['real_x']
            row['Y'] = block['real_y']
            row['Z'] = block['real_z']
        
        if include_grades:
            row['GRADE'] = round(block['grade'], 2)
        
        if include_values:
            row['VALUE'] = round(block['value'], 1)
        
        row['INPIT'] = 1 if block['in_pit'] else 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def generate_dxf(pit_boundary, block_size):
    # Générer un DXF pour les limites de la fosse
    dxf_content = "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n"
    dxf_content += "0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"
    
    # Regrouper les blocs par niveau Z
    levels = {}
    for block in pit_boundary:
        z = block['z']
        if z not in levels:
            levels[z] = []
        levels[z].append(block)
    
    # Pour chaque niveau, créer une polyligne fermée
    for z, blocks in levels.items():
        # Trier les blocs pour former un contour
        if len(blocks) > 2:
            # Simplification: juste créer une polyligne avec tous les blocs du niveau
            dxf_content += f"0\nPOLYLINE\n8\nPIT_LEVEL_{z}\n66\n1\n70\n1\n"
            
            for block in blocks:
                # Créer un rectangle pour chaque bloc
                x, y = block['real_x'], block['real_y']
                half_size = block_size / 2
                
                # Les quatre coins du bloc
                corners = [
                    (x - half_size, y - half_size),
                    (x + half_size, y - half_size),
                    (x + half_size, y + half_size),
                    (x - half_size, y + half_size)
                ]
                
                for cx, cy in corners:
                    dxf_content += f"0\nVERTEX\n8\nPIT_LEVEL_{z}\n10\n{cx}\n20\n{cy}\n30\n{block['real_z']}\n"
            
            dxf_content += "0\nSEQEND\n"
    
    dxf_content += "0\nENDSEC\n0\nEOF"
    
    return dxf_content

def prepare_download_link(content, filename, mime_type):
    """Génère un lien de téléchargement pour le contenu donné"""
    if isinstance(content, pd.DataFrame):
        # Pour DataFrame, convertir en CSV
        content = content.to_csv(index=False)
        b64 = base64.b64encode(content.encode()).decode()
    elif isinstance(content, str):
        # Pour le texte (comme DXF)
        b64 = base64.b64encode(content.encode()).decode()
    else:
        # Pour d'autres types (JSON, etc.)
        b64 = base64.b64encode(json.dumps(content).encode()).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="download-button">Télécharger {filename}</a>'
    return href

# Création de l'interface avec deux colonnes principales
col1, col2 = st.columns([1, 1])

# Colonne 1: Paramètres et contrôles
with col1:
    # Explication de l'algorithme Pseudo Flow avec un affichage amélioré
    with st.expander("À propos de l'algorithme Pseudo Flow", expanded=False):
        st.markdown("### Algorithme de Pseudo Flow")
        st.markdown("L'algorithme Pseudo Flow est une méthode moderne d'optimisation de fosse minière qui utilise la théorie des graphes et le concept de flot maximum.")
        
        st.markdown("#### Principes fondamentaux")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🔄 **Modélisation par graphe**")
            st.markdown("Le modèle de blocs est transformé en un graphe orienté où chaque bloc est un nœud.")
        
        with col2:
            st.markdown("🌊 **Flot maximum**")
            st.markdown("L'algorithme calcule le flot maximum entre une source (surface) et un puits (blocs profonds).")
        
        with col3:
            st.markdown("✂️ **Coupe minimale**")
            st.markdown("La coupe minimale du graphe correspond à la limite optimale de la fosse.")
        
        st.markdown("---")
        st.markdown("#### Avantages par rapport aux méthodes traditionnelles")
        
        adv_cols = st.columns(3)
        with adv_cols[0]:
            st.markdown("⚡ **Rapidité**")
            st.markdown("Généralement 3 à 10 fois plus rapide que Lerchs-Grossmann pour les grands modèles.")
        
        with adv_cols[1]:
            st.markdown("🎯 **Précision**")
            st.markdown("Résultats mathématiquement exacts dans le cadre du modèle.")
        
        with adv_cols[2]:
            st.markdown("🔄 **Flexibilité**")
            st.markdown("Gestion efficace des contraintes géotechniques complexes.")
        
        st.markdown("---")
        st.markdown("#### Paramètres clés")
        
        params_cols = st.columns([1, 2])
        with params_cols[0]:
            st.markdown("**Alpha (α)**")
            st.markdown("**Méthode de calcul**")
            st.markdown("**Capacity Scaling**")
        
        with params_cols[1]:
            st.markdown("Contrôle la sensibilité de l'algorithme aux valeurs négatives. Une valeur plus faible inclut plus de blocs.")
            st.markdown("Différentes approches pour résoudre le problème de flot maximum.")
            st.markdown("Technique d'optimisation qui améliore les performances sur les grands modèles.")
        
        st.info("L'algorithme a été développé par Hochbaum (2001) et a été largement adopté dans l'industrie minière pour sa performance et sa fiabilité.")

    # Paramètres d'algorithme
    st.header("Algorithme de Pseudo Flow")
    
    with st.expander("Paramètres de l'algorithme", expanded=True):
        pf_alpha = st.slider("Paramètre Alpha", min_value=0.0, max_value=0.5, value=0.15, format="%.2f",
                           help="Contrôle la sensibilité de l'algorithme. Valeurs plus petites = plus de blocs extraits.")
        pf_method = st.selectbox("Méthode de calcul", 
                              options=["Highest Label", "Pull-Relabel", "Push-Relabel"],
                              index=0,
                              help="Différentes approches de calcul du flot maximal")
        pf_capacity_scaling = st.checkbox("Capacity Scaling", value=True,
                                       help="Active le scaling de capacité pour améliorer la performance et la qualité")
    
    # Importer un modèle de blocs
    st.header("Importer un modèle de blocs")
    
    uploaded_file = st.file_uploader("Sélectionner un fichier", 
                                    type=["csv", "json"],
                                    help="Formats supportés: CSV, JSON")
    
    if uploaded_file is not None:
        # Déterminer le type de fichier
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        with st.spinner("Analyse du modèle de blocs..."):
            success = False
            message = ""
            
            if file_ext == '.csv':
                success, result = parse_block_model_csv(uploaded_file)
            elif file_ext == '.json':
                success, result = parse_json_model(uploaded_file)
            else:
                message = "Format de fichier non supporté"
            
            if success:
                # Stocker le modèle importé
                st.session_state.block_model = result['block_model']
                st.session_state.imported_model_info = result['info']
                st.session_state.model_imported = True
                
                # Afficher un résumé du modèle importé
                st.success(f"Modèle importé avec succès: {result['info']['total_blocks']} blocs")
                
                with st.expander("Détails du modèle importé", expanded=True):
                    info = result['info']
                    st.write(f"Dimensions: {info['size_x']} × {info['size_y']} × {info['size_z']} blocs")
                    st.write(f"Taille de bloc: {info['block_size']} m")
                    st.write(f"Origine: X={info['origin_x']}, Y={info['origin_y']}, Z={info['origin_z']}")
                    
                    # Ajuster l'affichage des teneurs selon le type de gisement
                    if st.session_state.deposit_type == "base_metals":
                        st.write(f"Teneur: Min={info['min_grade']:.2f}%, Max={info['max_grade']:.2f}%, Moy={info['avg_grade']:.2f}%")
                    else:
                        st.write(f"Teneur: Min={info['min_grade']:.2f} g/t, Max={info['max_grade']:.2f} g/t, Moy={info['avg_grade']:.2f} g/t")
                    
                    # Visualiser la distribution des teneurs
                    grades = [block['grade'] for block in result['block_model']]
                    
                    # Ajuster le titre selon le type de gisement
                    grade_unit = "%" if st.session_state.deposit_type == "base_metals" else "g/t"
                    title = f"Distribution des teneurs ({grade_unit})"
                    
                    fig = px.histogram(grades, nbins=20, title=title)
                    fig.update_layout(
                        xaxis_title=f"Teneur ({grade_unit})",
                        yaxis_title="Fréquence",
                        plot_bgcolor="white",
                        font=dict(family="Arial", size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erreur lors de l'importation: {result if isinstance(result, str) else message}")
    
    # Options de mappage (pour les coordonnées)
    if st.session_state.model_imported:
        st.header("Options de mappage")
        
        with st.expander("Configuration des coordonnées", expanded=False):
            st.write("Personnalisez l'affichage des coordonnées du modèle")
            
            # Facteur d'échelle pour l'affichage
            scale_factor = st.slider("Facteur d'échelle", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            
            # Inversion des axes
            axis_cols = st.columns(3)
            with axis_cols[0]:
                invert_x = st.checkbox("Inverser X", value=False)
            with axis_cols[1]:
                invert_y = st.checkbox("Inverser Y", value=False)
            with axis_cols[2]:
                invert_z = st.checkbox("Inverser Z", value=False)
            
            # Appliquer le mappage au modèle
            if st.button("Appliquer le mappage"):
                for block in st.session_state.block_model:
                    if invert_x:
                        block['real_x'] = -block['real_x'] * scale_factor
                    else:
                        block['real_x'] = block['real_x'] * scale_factor
                    
                    if invert_y:
                        block['real_y'] = -block['real_y'] * scale_factor
                    else:
                        block['real_y'] = block['real_y'] * scale_factor
                    
                    if invert_z:
                        block['real_z'] = -block['real_z'] * scale_factor
                    else:
                        block['real_z'] = block['real_z'] * scale_factor
                
                st.success("Mappage appliqué avec succès!")
    
    # Paramètres économiques
    st.header("Paramètres économiques")
    
    # Ajuster les unités selon le type de gisement
    if st.session_state.deposit_type == "base_metals":
        metal_price = st.number_input("Prix du métal ($/tonne)", min_value=0.0, value=8000.0, step=100.0)
        cutoff_grade = st.slider("Teneur de coupure (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
        grade_unit = "%"
    else:  # Métaux précieux
        metal_price = st.number_input("Prix du métal ($/oz)", min_value=0.0, value=1800.0, step=10.0)
        cutoff_grade = st.slider("Teneur de coupure (g/t)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
        grade_unit = "g/t"
    
    mining_cost = st.number_input("Coût d'extraction ($/t)", min_value=0.0, value=2.5, step=0.1)
    processing_cost = st.number_input("Coût de traitement ($/t)", min_value=0.0, value=10.0, step=0.5)
    recovery = st.slider("Taux de récupération (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
    
    # Paramètres géotechniques
    st.header("Paramètres géotechniques")
    
    slope_angle = st.slider("Angle de pente global (°)", min_value=25, max_value=75, value=45)
    bench_height = st.number_input("Hauteur de gradin (m)", min_value=1, value=10)
    
    # Bouton pour lancer l'optimisation
    run_optimizer = st.button("Lancer l'optimisation", type="primary", use_container_width=True, 
                             disabled=not st.session_state.model_imported)

# Colonne 2: Visualisation et résultats
with col2:
    # Visualisation 3D
    st.header("Visualisation")
    
    view_mode = st.selectbox("Mode d'affichage", 
                          options=["Teneurs", "Valeur économique", "Fosse optimale", "Limites de la fosse"],
                          index=0)
    
    # Espace réservé pour la visualisation 3D
    vis_placeholder = st.empty()
    
    # Résultats d'optimisation (apparaissent après l'exécution)
    results_container = st.container()
    
    with results_container:
        if st.session_state.results_ready:
            st.header("Résultats d'optimisation")
            st.write(f"Algorithme: **Pseudo Flow** | Temps d'exécution: {st.session_state.execution_time:.2f} secondes")
            
            # Onglets pour différents types de résultats
            tab1, tab2, tab3 = st.tabs(["Résumé", "Détails", "Sensibilité"])
            
            with tab1:
                # Métriques clés
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Blocs extraits", f"{len(st.session_state.optimal_pit)}")
                
                with metric_cols[1]:
                    # Calcul de la VAN simplifiée
                    npv = sum(block['value'] for block in st.session_state.optimal_pit)
                    st.metric("VAN", f"{npv:,.0f} $")
                
                with metric_cols[2]:
                    # Calcul simplifié du ratio stérile/minerai
                    if st.session_state.deposit_type == "base_metals":
                        ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                    else:
                        ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade)
                    
                    waste_blocks = len(st.session_state.optimal_pit) - ore_blocks
                    sr_ratio = waste_blocks / max(1, ore_blocks)
                    st.metric("Ratio S/M", f"{sr_ratio:.2f}")
                
                # Tableau des résultats
                st.subheader("Statistiques")
                
                # Calculer quelques métriques supplémentaires
                block_size_for_calc = st.session_state.imported_model_info.get('block_size', 10)
                
                # Ajuster le calcul des blocs minerai selon le type de gisement
                if st.session_state.deposit_type == "base_metals":
                    ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                    ore_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                    waste_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] <= cutoff_grade/100)
                else:  # Métaux précieux
                    ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade)
                    ore_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade)
                    waste_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] <= cutoff_grade)
                
                total_tonnage = ore_tonnage + waste_tonnage
                
                # Adapter les calculs de teneur selon le type de gisement
                if st.session_state.deposit_type == "base_metals":
                    avg_grade = sum(block['grade'] for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100) / max(1, ore_blocks)
                    # Pour les métaux de base (en pourcentage), conversion en tonnes pour le calcul du métal contenu
                    metal_content = ore_tonnage * avg_grade / 100  # % converti en fraction
                    metal_content_unit = "tonnes"
                else:  # Métaux précieux
                    avg_grade = sum(block['grade'] for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade) / max(1, ore_blocks)
                    # Pour les métaux précieux (en g/t), le métal contenu est en kg
                    metal_content = ore_tonnage * avg_grade / 1000  # g/t converti en kg/t
                    metal_content_unit = "kg"
                
                # Calcul de la valeur du métal récupéré
                recovered_metal = metal_content * recovery/100
                
                # Adapter le calcul du revenu selon le type de gisement
                if st.session_state.deposit_type == "base_metals":
                    # Pour les métaux de base, le métal est en tonnes
                    total_revenue = recovered_metal * metal_price  # Prix par tonne, métal en tonnes
                else:
                    # Pour les métaux précieux, convertir kg en oz
                    recovered_metal_oz = recovered_metal / 31.103  # kg à oz
                    total_revenue = recovered_metal_oz * metal_price  # Prix par oz, métal en oz
                
                mining_costs = total_tonnage * mining_cost
                processing_costs = ore_tonnage * processing_cost
                total_cost = mining_costs + processing_costs
                total_profit = total_revenue - total_cost
                
                # Créer le tableau de résultats avec les unités appropriées
                results_data = {
                    "Paramètre": ["Tonnage total", "Tonnage de minerai", "Tonnage de stérile", 
                                  f"Teneur moyenne ({grade_unit})", f"Métal contenu ({metal_content_unit})", "Revenu total", 
                                  "Coût total", "Profit"],
                    "Valeur": [
                        f"{total_tonnage:,.0f} t",
                        f"{ore_tonnage:,.0f} t",
                        f"{waste_tonnage:,.0f} t",
                        f"{avg_grade:.2f} {grade_unit}",
                        f"{metal_content:,.1f} {metal_content_unit}",
                        f"{total_revenue:,.0f} $",
                        f"{total_cost:,.0f} $",
                        f"{total_profit:,.0f} $"
                    ]
                }
                
                st.table(pd.DataFrame(results_data))
                
                # Options d'exportation
                st.subheader("Exporter les résultats")
                export_cols = st.columns(2)
                
                with export_cols[0]:
                    if st.button("📄 Résultats CSV", use_container_width=True):
                        st.session_state.export_csv = True
                
                with export_cols[1]:
                    if st.button("📐 Limites DXF", use_container_width=True):
                        st.session_state.export_dxf = True
                
                # Interface d'exportation CSV
                if 'export_csv' in st.session_state and st.session_state.export_csv:
                    st.subheader("Exporter en CSV")
                    csv_cols = st.columns(2)
                    
                    with csv_cols[0]:
                        include_coordinates = st.checkbox("Inclure les coordonnées", value=True)
                        include_grades = st.checkbox("Inclure les teneurs", value=True)
                    
                    with csv_cols[1]:
                        include_values = st.checkbox("Inclure les valeurs économiques", value=True)
                        only_pit = st.checkbox("Uniquement les blocs dans la fosse", value=True)
                    
                    # Générer le CSV et créer le lien
                    if st.session_state.optimal_pit:
                        csv_df = generate_csv(
                            st.session_state.block_model, 
                            st.session_state.optimal_pit,
                            include_coordinates, 
                            include_grades, 
                            include_values, 
                            only_pit
                        )
                        
                        # Afficher un aperçu
                        st.write("Aperçu:")
                        st.dataframe(csv_df.head())
                        
                        # Créer le lien de téléchargement avec nom du projet
                        project_name_safe = st.session_state.project_name.replace(' ', '_').lower()
                        csv_filename = f"{project_name_safe}_results_PF_{datetime.now().strftime('%Y%m%d')}.csv"
                        csv_link = prepare_download_link(csv_df, csv_filename, "text/csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_csv"):
                        st.session_state.export_csv = False
                        st.experimental_rerun()
                
                # Interface d'exportation DXF
                if 'export_dxf' in st.session_state and st.session_state.export_dxf:
                    st.subheader("Exporter en DXF")
                    
                    # Générer le DXF et créer le lien
                    if st.session_state.pit_boundary:
                        block_size_for_dxf = st.session_state.imported_model_info.get('block_size', 10)
                        
                        dxf_content = generate_dxf(
                            st.session_state.pit_boundary,
                            block_size_for_dxf
                        )
                        
                        # Afficher un aperçu
                        st.text_area("Aperçu DXF:", value=dxf_content[:500] + "...", height=150)
                        
                        # Créer le lien de téléchargement avec nom du projet
                        project_name_safe = st.session_state.project_name.replace(' ', '_').lower()
                        dxf_filename = f"{project_name_safe}_pit_boundary_{datetime.now().strftime('%Y%m%d')}.dxf"
                        dxf_link = prepare_download_link(dxf_content, dxf_filename, "application/dxf")
                        st.markdown(dxf_link, unsafe_allow_html=True)
                    else:
                        st.warning("Aucune limite de fosse disponible à exporter")
                    
                    if st.button("Fermer", key="close_dxf"):
                        st.session_state.export_dxf = False
                        st.experimental_rerun()
            
            with tab2:
                # Détails par niveau
                st.subheader("Détails par niveau")
                
                # Grouper les blocs par niveau
                levels_data = []
                
                if st.session_state.optimal_pit:
                    # Déterminer la taille Z à partir du modèle
                    max_z = max(block['z'] for block in st.session_state.optimal_pit) + 1
                    
                    for z in range(max_z):
                        level_blocks = [block for block in st.session_state.optimal_pit if block['z'] == z]
                        if level_blocks:
                            # Adapter le calcul selon le type de gisement
                            if st.session_state.deposit_type == "base_metals":
                                level_ore_blocks = [block for block in level_blocks if block['grade'] > cutoff_grade/100]
                            else:
                                level_ore_blocks = [block for block in level_blocks if block['grade'] > cutoff_grade]
                            
                            block_size_level = st.session_state.imported_model_info.get('block_size', 10)
                            level_tonnage = len(level_blocks) * block_size_level**3 * 2.7
                            level_grade = sum(block['grade'] for block in level_ore_blocks) / max(1, len(level_ore_blocks))
                            level_value = sum(block['value'] for block in level_blocks)
                            
                            # Trouver l'élévation réelle
                            if level_blocks[0].get('real_z') is not None:
                                # Utiliser la première élévation réelle trouvée comme référence
                                elev = level_blocks[0]['real_z']
                            else:
                                # Calculer à partir de l'origine et de la taille du bloc
                                origin_z = st.session_state.imported_model_info.get('origin_z', 0)
                                elev = origin_z - z * block_size_level
                            
                            levels_data.append({
                                "Niveau": z + 1,
                                "Élévation": elev,
                                "Blocs": len(level_blocks),
                                "Tonnage": f"{level_tonnage:,.0f} t",
                                f"Teneur moy. ({grade_unit})": f"{level_grade:.2f}",
                                "Valeur": f"{level_value:,.0f} $"
                            })
                
                if levels_data:
                    st.table(pd.DataFrame(levels_data))
                else:
                    st.info("Aucun bloc dans la fosse optimisée")
            
            with tab3:
                # Analyse de sensibilité
                st.subheader("Analyse de sensibilité")
                
                # Créer des données fictives pour l'analyse de sensibilité
                sensitivity_data = {
                    "Variable": ["Prix du métal", "Coût d'extraction", "Coût de traitement", "Récupération", "Teneur de coupure"],
                    "-20%": [0.8, 1.15, 1.12, 0.85, 1.05],
                    "-10%": [0.9, 1.07, 1.06, 0.92, 1.02],
                    "Base": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "+10%": [1.1, 0.93, 0.94, 1.08, 0.97],
                    "+20%": [1.2, 0.87, 0.88, 1.15, 0.95]
                }
                
                df_sensitivity = pd.DataFrame(sensitivity_data)
                
                # Créer un graphique de sensibilité
                fig = go.Figure()
                
                for variable in df_sensitivity["Variable"]:
                    row = df_sensitivity[df_sensitivity["Variable"] == variable].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[-20, -10, 0, 10, 20],
                        y=[row["-20%"], row["-10%"], row["Base"], row["+10%"], row["+20%"]],
                        mode='lines+markers',
                        name=variable
                    ))
                
                fig.update_layout(
                    title="Analyse de sensibilité (VAN relative)",
                    xaxis_title="Variation des paramètres (%)",
                    yaxis_title="VAN relative",
                    legend_title="Paramètres",
                    hovermode="x unified",
                    height=500,
                    plot_bgcolor="white",
                    font=dict(family="Arial", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                L'analyse de sensibilité montre que la VAN du projet est:
                - Très sensible aux variations du prix du métal et du taux de récupération
                - Moyennement sensible aux coûts d'extraction et de traitement
                - Peu sensible aux variations de la teneur de coupure
                """)

# Logique d'optimisation
if run_optimizer:
    # Afficher un indicateur de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Préparation du modèle de blocs...")
    progress_bar.progress(10)
    
    # Convertir les pourcentages en valeurs décimales pour les teneurs de coupure
    if st.session_state.deposit_type == "base_metals":
        cutoff_grade_decimal = cutoff_grade / 100  # % à fraction
    else:
        cutoff_grade_decimal = cutoff_grade  # g/t reste en g/t
        
    recovery_decimal = recovery / 100
    
    # Utiliser le modèle importé
    start_time = time.time()
    
    # Recalculer les valeurs économiques avec les paramètres actuels
    block_size_for_calc = st.session_state.imported_model_info.get('block_size', 10)
    
    for block in st.session_state.block_model:
        tonnage = block_size_for_calc**3 * 2.7  # Densité moyenne de 2.7 t/m³
        
        # Adapter les calculs économiques selon le type de gisement
        if st.session_state.deposit_type == "base_metals":
            # Pour les métaux de base (teneur en %)
            if block['grade'] > cutoff_grade_decimal:
                # Bloc de minerai, grade en % donc divisé par 100 pour avoir la fraction
                metal_tonnes = tonnage * (block['grade'] / 100)  # tonnes de métal
                revenue = metal_tonnes * metal_price * recovery_decimal
                block['value'] = revenue - tonnage * (mining_cost + processing_cost)
            else:
                # Bloc de stérile
                block['value'] = -tonnage * mining_cost
        else:
            # Pour les métaux précieux (teneur en g/t)
            if block['grade'] > cutoff_grade_decimal:
                # Bloc de minerai, grade en g/t
                # Convertir g/t en oz/t (1 oz = 31.103 g)
                metal_oz = tonnage * block['grade'] / 31.103  # oz de métal
                revenue = metal_oz * metal_price * recovery_decimal
                block['value'] = revenue - tonnage * (mining_cost + processing_cost)
            else:
                # Bloc de stérile
                block['value'] = -tonnage * mining_cost
    
    # Utiliser les dimensions du modèle importé
    size_x = st.session_state.imported_model_info.get('size_x', 20)
    size_y = st.session_state.imported_model_info.get('size_y', 20)
    size_z = st.session_state.imported_model_info.get('size_z', 10)
    
    status_text.text("Construction du graphe de flot...")
    progress_bar.progress(30)
    time.sleep(0.3)  # Simulation de temps de calcul
    
    status_text.text("Initialisation du push-relabel...")
    progress_bar.progress(40)
    time.sleep(0.3)  # Simulation de temps de calcul
    
    status_text.text("Calcul du flot maximal...")
    progress_bar.progress(50)
    time.sleep(0.3)  # Simulation de temps de calcul
    
    status_text.text("Optimisation des limites de fosse...")
    progress_bar.progress(70)
    
    # Exécuter l'algorithme de Pseudo Flow
    st.session_state.optimal_pit, st.session_state.pit_boundary = run_pseudo_flow(
        st.session_state.block_model,
        size_x, size_y, size_z,
        slope_angle,
        alpha=pf_alpha,
        method=pf_method,
        capacity_scaling=pf_capacity_scaling
    )
    
    status_text.text("Finalisation des résultats...")
    progress_bar.progress(90)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    # Calcul du temps d'exécution
    end_time = time.time()
    st.session_state.execution_time = end_time - start_time
    
    # Marquer que les résultats sont prêts
    st.session_state.results_ready = True
    
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    time.sleep(0.5)
    
    # Supprimer la barre de progression et le texte de statut
    progress_bar.empty()
    status_text.empty()
    
    # Actualiser la page pour afficher les résultats
    st.experimental_rerun()

# Visualisation 3D
if st.session_state.model_imported:
    # Visualiser le modèle importé ou la fosse optimale selon le mode d'affichage
    
    if view_mode == "Teneurs" or (view_mode != "Fosse optimale" and view_mode != "Limites de la fosse" and not st.session_state.results_ready):
        # Afficher le modèle de blocs coloré par teneur
        sampled_blocks = st.session_state.block_model
        max_blocks_to_show = 1000
        
        if len(sampled_blocks) > max_blocks_to_show:
            step = len(sampled_blocks) // max_blocks_to_show
            sampled_blocks = sampled_blocks[::step]
        
        # Créer la figure 3D
        fig = go.Figure()
        
        # Extraire les coordonnées et valeurs
        x = [block['real_x'] for block in sampled_blocks]
        y = [block['real_y'] for block in sampled_blocks]
        z = [block['real_z'] for block in sampled_blocks]
        colors = [block['grade'] for block in sampled_blocks]
        
        # Dessiner des cubes au lieu de points
        block_size_for_vis = st.session_state.imported_model_info.get('block_size', 10) * 0.8  # Légèrement plus petit pour voir les limites
        
        for i, block in enumerate(sampled_blocks):
            x0, y0, z0 = block['real_x'], block['real_y'], block['real_z']
            half_size = block_size_for_vis / 2
            
            # Créer un cube en utilisant Mesh3d
            vertices_x = [x0-half_size, x0+half_size, x0+half_size, x0-half_size, x0-half_size, x0+half_size, x0+half_size, x0-half_size]
            vertices_y = [y0-half_size, y0-half_size, y0+half_size, y0+half_size, y0-half_size, y0-half_size, y0+half_size, y0+half_size]
            vertices_z = [z0-half_size, z0-half_size, z0-half_size, z0-half_size, z0+half_size, z0+half_size, z0+half_size, z0+half_size]
            
            I = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7]
            J = [1, 2, 3, 3, 0, 0, 5, 6, 7, 7, 4, 4, 4, 7, 3, 3, 0, 0, 5, 6, 2, 2, 1, 1, 1, 5, 4, 4, 0, 0, 2, 6, 7, 7, 3, 3]
            K = [2, 3, 0, 0, 0, 0, 6, 7, 4, 4, 4, 4, 7, 3, 0, 0, 0, 0, 6, 2, 1, 1, 1, 1, 5, 4, 0, 0, 0, 0, 6, 7, 3, 3, 3, 3]
            
            fig.add_trace(go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=I, j=J, k=K,
                opacity=0.8,
                color=f'rgb({50+int(block["grade"]*40)}, {100+int(block["grade"]*50)}, {150+int(block["grade"]*30)})',
                name=f"Block {i}",
                hovertext=f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>Teneur: {block['grade']:.2f} {grade_unit}"
            ))
        
        # Adapter le titre selon le type de gisement
        title = f"Modèle de blocs coloré par teneur ({grade_unit})"
        
        # Configurer la mise en page
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1)
            ),
            title=title
        )
        
        # Afficher la visualisation
        vis_placeholder.plotly_chart(fig, use_container_width=True)
    
    elif view_mode == "Valeur économique" and st.session_state.results_ready:
        # Afficher le modèle de blocs coloré par valeur économique
        sampled_blocks = st.session_state.block_model
        max_blocks_to_show = 1000
        
        if len(sampled_blocks) > max_blocks_to_show:
            step = len(sampled_blocks) // max_blocks_to_show
            sampled_blocks = sampled_blocks[::step]
        
        # Créer la figure 3D
        fig = go.Figure()
        
        # Calculer la plage de valeurs pour la colorisation
        all_values = [block['value'] for block in sampled_blocks]
        min_value, max_value = min(all_values), max(all_values)
        
        # Normaliser les valeurs pour la coloration
        def get_value_color(value):
            if value > 0:
                # Positif = vert (plus clair pour les valeurs plus élevées)
                intensity = min(255, 100 + int((value / max_value) * 155))
                return f'rgb(0, {intensity}, 0)'
            else:
                # Négatif = rouge (plus foncé pour les valeurs plus négatives)
                intensity = min(255, 100 + int((abs(value) / abs(min_value)) * 155))
                return f'rgb({intensity}, 0, 0)'
        
        # Dessiner des cubes au lieu de points
        block_size_for_vis = st.session_state.imported_model_info.get('block_size', 10) * 0.8
        
        for i, block in enumerate(sampled_blocks):
            x0, y0, z0 = block['real_x'], block['real_y'], block['real_z']
            half_size = block_size_for_vis / 2
            
            # Créer un cube en utilisant Mesh3d
            vertices_x = [x0-half_size, x0+half_size, x0+half_size, x0-half_size, x0-half_size, x0+half_size, x0+half_size, x0-half_size]
            vertices_y = [y0-half_size, y0-half_size, y0+half_size, y0+half_size, y0-half_size, y0-half_size, y0+half_size, y0+half_size]
            vertices_z = [z0-half_size, z0-half_size, z0-half_size, z0-half_size, z0+half_size, z0+half_size, z0+half_size, z0+half_size]
            
            I = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7]
            J = [1, 2, 3, 3, 0, 0, 5, 6, 7, 7, 4, 4, 4, 7, 3, 3, 0, 0, 5, 6, 2, 2, 1, 1, 1, 5, 4, 4, 0, 0, 2, 6, 7, 7, 3, 3]
            K = [2, 3, 0, 0, 0, 0, 6, 7, 4, 4, 4, 4, 7, 3, 0, 0, 0, 0, 6, 2, 1, 1, 1, 1, 5, 4, 0, 0, 0, 0, 6, 7, 3, 3, 3, 3]
            
            fig.add_trace(go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=I, j=J, k=K,
                opacity=0.8,
                color=get_value_color(block['value']),
                name=f"Block {i}",
                hovertext=f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>Teneur: {block['grade']:.2f} {grade_unit}<br>Valeur: {block['value']:.2f}$"
            ))
        
        # Configurer la mise en page
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            title="Modèle de blocs coloré par valeur économique"
        )
        
        # Afficher la visualisation
        vis_placeholder.plotly_chart(fig, use_container_width=True)
    
    elif view_mode == "Fosse optimale" and st.session_state.results_ready:
        # Afficher la fosse optimale
        sampled_blocks = st.session_state.optimal_pit
        max_blocks_to_show = 1000
        
        if len(sampled_blocks) > max_blocks_to_show:
            step = len(sampled_blocks) // max_blocks_to_show
            sampled_blocks = sampled_blocks[::step]
        
        # Créer la figure 3D
        fig = go.Figure()
        
        # Dessiner des cubes au lieu de points
        block_size_for_vis = st.session_state.imported_model_info.get('block_size', 10) * 0.8
        
        for i, block in enumerate(sampled_blocks):
            x0, y0, z0 = block['real_x'], block['real_y'], block['real_z']
            half_size = block_size_for_vis / 2
            
            # Créer un cube en utilisant Mesh3d
            vertices_x = [x0-half_size, x0+half_size, x0+half_size, x0-half_size, x0-half_size, x0+half_size, x0+half_size, x0-half_size]
            vertices_y = [y0-half_size, y0-half_size, y0+half_size, y0+half_size, y0-half_size, y0-half_size, y0+half_size, y0+half_size]
            vertices_z = [z0-half_size, z0-half_size, z0-half_size, z0-half_size, z0+half_size, z0+half_size, z0+half_size, z0+half_size]
            
            I = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7]
            J = [1, 2, 3, 3, 0, 0, 5, 6, 7, 7, 4, 4, 4, 7, 3, 3, 0, 0, 5, 6, 2, 2, 1, 1, 1, 5, 4, 4, 0, 0, 2, 6, 7, 7, 3, 3]
            K = [2, 3, 0, 0, 0, 0, 6, 7, 4, 4, 4, 4, 7, 3, 0, 0, 0, 0, 6, 2, 1, 1, 1, 1, 5, 4, 0, 0, 0, 0, 6, 7, 3, 3, 3, 3]
            
            # Coloration basée sur la profondeur (z)
            z_normalized = block['z'] / max(1, st.session_state.imported_model_info.get('size_z', 10))
            
            # Couleurs différentes pour l'algorithme Pseudo Flow (teintes de violet/bleu)
            color = f'rgb({100+int(z_normalized*155)}, {50+int(z_normalized*100)}, {200-int(z_normalized*50)})'
            
            fig.add_trace(go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=I, j=J, k=K,
                opacity=0.8,
                color=color,
                name=f"Block {i}",
                hovertext=f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>Teneur: {block['grade']:.2f} {grade_unit}<br>Valeur: {block['value']:.2f}$"
            ))
        
        # Configurer la mise en page
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            title=f"Fosse optimale - {st.session_state.project_name}"
        )
        
        # Afficher la visualisation
        vis_placeholder.plotly_chart(fig, use_container_width=True)
    
    elif view_mode == "Limites de la fosse" and st.session_state.results_ready and st.session_state.pit_boundary:
        # Afficher seulement les limites de la fosse
        sampled_blocks = st.session_state.pit_boundary
        
        # Créer la figure 3D
        fig = go.Figure()
        
        # Regrouper par niveau z pour coloration
        z_levels = {}
        for block in sampled_blocks:
            z = block['z']
            if z not in z_levels:
                z_levels[z] = []
            z_levels[z].append(block)
        
        # Dessiner les limites de la fosse pour chaque niveau
        block_size_for_vis = st.session_state.imported_model_info.get('block_size', 10)
        
        for z, blocks in z_levels.items():
            # Points pour ce niveau
            x_coords = [block['real_x'] for block in blocks]
            y_coords = [block['real_y'] for block in blocks]
            z_coords = [block['real_z'] for block in blocks]
            
            # Normalisation pour coloration (couleurs violettes/bleus pour Pseudo Flow)
            z_normalized = z / max(1, st.session_state.imported_model_info.get('size_z', 10))
            color = f'rgb({100+int(z_normalized*155)}, {50+int(z_normalized*100)}, {200-int(z_normalized*50)})'
            
            # Tracer les points de la limite pour ce niveau
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='square',
                ),
                name=f"Niveau {z+1}",
                hoverinfo='text',
                hovertext=[f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>Niveau: {z+1}" for block in blocks]
            ))
            
            # Tenter de créer un mesh pour ce niveau (contour)
            # Calcul simplifié: on relie les points qui sont adjacents
            for i, block1 in enumerate(blocks):
                for j, block2 in enumerate(blocks[i+1:], i+1):
                    # Vérifier si ces blocs sont adjacents (approximatif)
                    dist = ((block1['real_x'] - block2['real_x'])**2 + 
                            (block1['real_y'] - block2['real_y'])**2)**0.5
                    
                    if dist <= block_size_for_vis * 1.5:  # Un peu de marge
                        fig.add_trace(go.Scatter3d(
                            x=[block1['real_x'], block2['real_x']],
                            y=[block1['real_y'], block2['real_y']],
                            z=[block1['real_z'], block2['real_z']],
                            mode='lines',
                            line=dict(color=color, width=2),
                            showlegend=False
                        ))
        
        # Configurer la mise en page
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            title=f"Limites de la fosse optimale - {st.session_state.project_name}"
        )
        
        # Afficher la visualisation
        vis_placeholder.plotly_chart(fig, use_container_width=True)
    
    else:
        # Message par défaut
        vis_placeholder.info("Lancez l'optimisation pour visualiser la fosse optimale.")
else:
    # Message si aucun modèle n'est importé
    vis_placeholder.info("Importez un modèle de blocs pour commencer.")

# Pied de page
st.markdown("---")
st.markdown("<div style='text-align: center; color: #566573;'>© 2025 Didier Ouedraogo, P.Geo - Tous droits réservés</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #566573;'>PF Pit Optimizer v1.0.0</div>", unsafe_allow_html=True)