"""Explicit metric boundaries, subordinate to original task requirements."""
from copy import deepcopy

BOUNDARIES = {
    'identity_consistency_score': ('appearance_identity', 'Visible person identity and appearance across frames.',
        'Wrong action, incorrect object possession, missing cuts, or failure to follow a narrative role alone.'),
    'identity_continuity': ('appearance_identity', 'Whether the same source-bound people persist across views.',
        'A correctly identified person performing the wrong action. Score that under action/state criteria.'),
    'clothing_color_score': ('appearance_attributes', 'Declared clothing attributes and their allowed changes.',
        'Action success, object custody, background layout, or desired plot outcome.'),
    'background_preservation_score': ('scene_preservation', 'Preservation of background features where required, allowing declared scene changes.',
        'A missing task prop or failed action alone; absence of a reference object is not proof a background was rebuilt.'),
    'scene_geometry': ('spatial_consistency', 'Coherence of visible fixed geometry and spatial relationships across views.',
        'A missing task prop alone. Do not infer an unseen till or other object from the desired story.'),
    'visual_quality': ('perceptual_quality', 'Visible blur, artifacts, detail, exposure, and other declared perceptual defects.',
        'Whether the desired story occurred, who holds a prop, or whether a narrative action is correct.'),
    'motion_coherence': ('physical_motion', 'Visible physical trajectories, contact, deformation and temporal coherence.',
        'Failure to perform the desired narrative. Occlusion alone is not teleportation.'),
    'action_alignment_score': ('task_actions', 'Required actors, actions, objects, direction, coverage and order.',
        'Attractive appearance or stable backgrounds do not compensate for an incorrect action.'),
}


def apply_boundaries(criteria):
    out = deepcopy(criteria)
    for name, rule in out.items():
        if name in BOUNDARIES and isinstance(rule, dict):
            dimension, measure, exclude = BOUNDARIES[name]
            rule['measurement_boundary'] = {'dimension': dimension, 'measure': measure, 'do_not_substitute': exclude,
                'precedence': 'The original criterion description remains authoritative. If it explicitly '
                    'requires a prop/attribute/compound condition, assess that clause too. These boundaries '
                    'do not delete original obligations, add requirements, or supply visual evidence.'}
    return out
