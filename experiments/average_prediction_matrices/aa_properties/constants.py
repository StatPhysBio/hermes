
HYDROPHOBIC_PROPS = [f'V{i}' for i in range(1, 19)]
STERIC_PROPS = [f'V{i}' for i in range(19, 36)]
ELECTRONIC_PROPS = [f'V{i}' for i in range(36, 51)]
PROP_TYPE_TO_PROPS = {
    'electronic': ELECTRONIC_PROPS,
    'hydrophobic': HYDROPHOBIC_PROPS,
    'steric': STERIC_PROPS,
}
PROP_TYPES = list(PROP_TYPE_TO_PROPS.keys())
ALL_PROPS = HYDROPHOBIC_PROPS + STERIC_PROPS + ELECTRONIC_PROPS

PROP_TYPE_TO_COLOR = {
    'electronic': 'orange',
    'hydrophobic': 'green',
    'steric': 'purple',
}
