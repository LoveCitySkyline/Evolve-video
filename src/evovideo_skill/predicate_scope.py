"""Compile judgment boundaries from structured sources, without parsing scenes."""
VERSION = "source-predicate-scope-v1"


def predicate_scope(source):
    path = source['source']
    result = {"protocol": VERSION, "scored_proposition": source['proposition'],
        "scope": "exact_source", "rule": "Judge only the quoted proposition. Other requirements "
            "are assessed separately. Context can resolve identity and references but cannot add requirements."}
    for field in ('preconditions', 'postconditions', 'invariants'):
        marker = '.' + field + '.'
        if marker in path:
            result.update(scope="state_attribute", state_key=path.split(marker, 1)[1],
                rule="Judge this state key against its complete declared value at the required time. "
                     "Do not import values of other state keys. A form/shape requirement is not a location "
                     "requirement unless the quoted value explicitly includes location; apply the same "
                     "separation to holder, contents, color and other attributes. Cite a visible incompatible "
                     "value of THIS property to refute it. Occlusion remains unknown.")
            return result
    if path == 'semantics.initial_setup':
        result.update(scope="initial_setup_only", rule="Judge only this setup sentence. Do not import "
            "additional first-shot preconditions, later actions or states. Respect its explicit quantity "
            "and modifiers, but do not turn a specified object into a ban on all unrelated objects.")
    elif '.events.' in path:
        result.update(scope="action_only", rule="Judge the declared actor, action, object and explicit "
            "qualifiers in this source. A wrong starting state or an extra earlier action does not alone "
            "prove this action did not occur. If the required action visibly occurs later WITHIN this "
            "window, assess it on that evidence; assess preconditions and state flow separately. "
            "Retain all explicit before/after/while/without requirements and obligation context.")
    return result
