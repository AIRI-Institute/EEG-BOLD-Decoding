from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

ALL_SUBJECTS = ['04', '07', '10', '11', '12', '13', '14', '15', '16', '19', '22']
EEG_SAMPLING_RATE = 250
FMRI_TR = 2.1  # sec

RUNS = [
    ('01', 'checker'),
    ('01', 'dme_run-01'),
    ('01', 'dme_run-02'),
    ('01', 'inscapes'),
    ('01', 'monkey1_run-01'),
    ('01', 'monkey1_run-02'),
    ('01', 'rest'),
    ('01', 'tp_run-01'),
    ('01', 'tp_run-02'),
    ('02', 'checker'),
    ('02', 'dmh_run-01'),
    ('02', 'dmh_run-02'),
    ('02', 'inscapes'),
    ('02', 'monkey2_run-01'),
    ('02', 'monkey2_run-02'),
    ('02', 'monkey5_run-01'),
    ('02', 'monkey5_run-02'),
    ('02', 'rest')
]

# IN SAMPLES WITH 2 Hz SAMPLING RATE. FOR SECONDS DIVIDE BY 2
TEST_SIZES = {
    'checker': 100,
    'dme': 250,
    'inscapes': 250,
    'monkey1': 150,
    'rest': 250,
    'tp': 100,
    'dmh': 250,
    'monkey2': 150,
    'monkey5': 150
}

# NOT USED CHANNELS: ECG, EOGL, EOGU
EEG_CHANNELS = [
    'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',
    'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
    'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4',
    'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP10', 'TP7', 'TP8',
    'TP9'
]

SUBCORT_ROIS = [
    ' Left Thalamus',
    ' Left Caudate',
    ' Left Putamen',
    ' Left Pallidum',
    ' Left Hippocampus',
    ' Left Amygdala',
    ' Left Accumbens',
    ' Right Thalamus',
    ' Right Caudate',
    ' Right Putamen',
    ' Right Pallidum',
    ' Right Hippocampus',
    ' Right Amygdala',
    ' Right Accumbens'
]

CORT_ROIS = [
    ' Left Frontal Pole', ' Right Frontal Pole', ' Left Insular Cortex', ' Right Insular Cortex',
    ' Left Superior Frontal Gyrus', ' Right Superior Frontal Gyrus', ' Left Middle Frontal Gyrus',
    ' Right Middle Frontal Gyrus', ' Left Inferior Frontal Gyrus in pars triangularis',
    ' Right Inferior Frontal Gyrus in pars triangularis',
    ' Left Inferior Frontal Gyrus in pars opercularis',
    ' Right Inferior Frontal Gyrus in pars opercularis', ' Left Precentral Gyrus',
    ' Right Precentral Gyrus', ' Left Temporal Pole', ' Right Temporal Pole',
    ' Left Superior Temporal Gyrus in anterior division',
    ' Right Superior Temporal Gyrus in anterior division',
    ' Left Superior Temporal Gyrus in posterior division',
    ' Right Superior Temporal Gyrus in posterior division',
    ' Left Middle Temporal Gyrus in anterior division',
    ' Right Middle Temporal Gyrus in anterior division',
    ' Left Middle Temporal Gyrus in posterior division',
    ' Right Middle Temporal Gyrus in posterior division',
    ' Left Middle Temporal Gyrus in temporooccipital part',
    ' Right Middle Temporal Gyrus in temporooccipital part',
    ' Left Inferior Temporal Gyrus in anterior division',
    ' Right Inferior Temporal Gyrus in anterior division',
    ' Left Inferior Temporal Gyrus in posterior division',
    ' Right Inferior Temporal Gyrus in posterior division',
    ' Left Inferior Temporal Gyrus in temporooccipital part',
    ' Right Inferior Temporal Gyrus in temporooccipital part', ' Left Postcentral Gyrus',
    ' Right Postcentral Gyrus', ' Left Superior Parietal Lobule', ' Right Superior Parietal Lobule',
    ' Left Supramarginal Gyrus in anterior division', ' Right Supramarginal Gyrus in anterior division',
    ' Left Supramarginal Gyrus in posterior division',
    ' Right Supramarginal Gyrus in posterior division', ' Left Angular Gyrus', ' Right Angular Gyrus',
    ' Left Lateral Occipital Cortex in superior division',
    ' Right Lateral Occipital Cortex in superior division',
    ' Left Lateral Occipital Cortex in inferior division',
    ' Right Lateral Occipital Cortex in inferior division', ' Left Intracalcarine Cortex',
    ' Right Intracalcarine Cortex', ' Left Frontal Medial Cortex', ' Right Frontal Medial Cortex',
    ' Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
    ' Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
    ' Left Subcallosal Cortex', ' Right Subcallosal Cortex', ' Left Paracingulate Gyrus',
    ' Right Paracingulate Gyrus', ' Left Cingulate Gyrus in anterior division',
    ' Right Cingulate Gyrus in anterior division', ' Left Cingulate Gyrus in posterior division',
    ' Right Cingulate Gyrus in posterior division', ' Left Precuneous Cortex',
    ' Right Precuneous Cortex', ' Left Cuneal Cortex', ' Right Cuneal Cortex',
    ' Left Frontal Orbital Cortex', ' Right Frontal Orbital Cortex',
    ' Left Parahippocampal Gyrus in anterior division',
    ' Right Parahippocampal Gyrus in anterior division',
    ' Left Parahippocampal Gyrus in posterior division',
    ' Right Parahippocampal Gyrus in posterior division', ' Left Lingual Gyrus', ' Right Lingual Gyrus',
    ' Left Temporal Fusiform Cortex in anterior division',
    ' Right Temporal Fusiform Cortex in anterior division',
    ' Left Temporal Fusiform Cortex in posterior division',
    ' Right Temporal Fusiform Cortex in posterior division', ' Left Temporal Occipital Fusiform Cortex',
    ' Right Temporal Occipital Fusiform Cortex', ' Left Occipital Fusiform Gyrus',
    ' Right Occipital Fusiform Gyrus', ' Left Frontal Opercular Cortex',
    ' Right Frontal Opercular Cortex', ' Left Central Opercular Cortex',
    ' Right Central Opercular Cortex', ' Left Parietal Opercular Cortex',
    ' Right Parietal Opercular Cortex', ' Left Planum Polare', ' Right Planum Polare',
    " Left Heschl's Gyrus (includes H1 and H2)", " Right Heschl's Gyrus (includes H1 and H2)",
    ' Left Planum Temporale', ' Right Planum Temporale', ' Left Supracalcarine Cortex',
    ' Right Supracalcarine Cortex', ' Left Occipital Pole', ' Right Occipital Pole'
]
