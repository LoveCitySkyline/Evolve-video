import unittest
from evovideo_skill.predicate_scope import predicate_scope


class PredicateScopeTests(unittest.TestCase):
    def test_properties_are_selected_by_source_not_keyword_in_target(self):
        for key, value in [('clay.form','flat spiral'), ('clay.location','mat'),
                           ('baton.holder','A'),('tray.contents','bread'),('novel.attribute','novel value')]:
            proposition = f'{key} equals {value!r}'
            result = predicate_scope({'source':f'shots[1].postconditions.{key}','proposition':proposition})
            self.assertEqual(result['state_key'],key)
            self.assertEqual(result['scored_proposition'],proposition)

    def test_setup_never_imports_other_properties_and_action_keeps_qualifiers(self):
        result = predicate_scope({'source':'semantics.initial_setup',
                                  'proposition':'A has one long clay rope on a pottery mat.'})
        self.assertEqual(result['scope'],'initial_setup_only')
        self.assertNotIn('straight',result['scored_proposition'])
        for text in ['A raises outer coils into a wall','A lifts the lid without moving the box',
                     'B takes the handle before A releases it']:
            result = predicate_scope({'source':'shots[1].events.event.obligations[0]','proposition':text})
            self.assertEqual(result['scope'],'action_only')
            self.assertEqual(result['scored_proposition'],text)
