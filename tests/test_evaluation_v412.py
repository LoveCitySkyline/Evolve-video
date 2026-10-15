"""Compile actor-qualified state questions; never pretend to test visual truth."""
from copy import deepcopy
import unittest

from evovideo_skill.isolated_observations import (
    VERSION, LEGACY_VERSION, BOUNDARY_VERSION, ACTOR_VERSION, SYSTEMS, qualify_state_actors)
from evovideo_skill.review_observations import compile_plan, wire_contract
from evovideo_skill.verifier_facts import citation_contract
from evovideo_skill.verifier_identity import registry
from evovideo_skill.research_subgraphs import stable_hash
from test_verifier_identity import SOURCE
from test_scoped_judgment import SPANS
import test_evaluation_v41 as isolation_fixture
from test_verifier_facts import POST_FACT


class ActorQualifiedStateTests(unittest.TestCase):
    def setUp(self):
        self.actors = registry({'metadata': {'h3_global_constraints': SOURCE}})

    def question(self, text, domain='state_attribute'):
        q = {'observation_domain': domain, 'proposition': text}
        qualify_state_actors(q, self.actors)
        return q

    def test_holder_body_parts_containers_and_actor_properties_are_explicit(self):
        for text, expected in [
            ("token.location equals 'A hand'", "token.location equals 'ACTOR[A] hand'"),
            ("tray.location equals 'B hands over water'", "tray.location equals 'ACTOR[B] hands over water'"),
            ("baton.holder equals 'A'", "baton.holder equals 'ACTOR[A]'"),
            ("A.location equals 'near B'", "ACTOR[A].location equals 'near ACTOR[B]'"),
            ("object.location equals 'B tray'", "object.location equals 'ACTOR[B] tray'"),
            ("object.location equals 'in A hands'", "object.location equals 'in ACTOR[A] hands'")]:
            with self.subTest(text=text):
                q = self.question(text)
                self.assertEqual(q['proposition'], expected)
                for a, spec in q['actor_references'].items():
                    self.assertEqual(spec['appearance_id'], self.actors[a]['appearance_id'])

    def test_undeclared_shape_labels_articles_and_action_prose_are_unchanged(self):
        for text, domain in [("ribbon.form equals 'V shape'", 'state_attribute'),
                             ("token.location equals 'a hand'", 'state_attribute'),
                             ('A customer holds one token.', 'initial_setup_only'),
                             ('A places the token in a tray.', 'action_only')]:
            self.assertEqual(self.question(text, domain), {'proposition': text, 'observation_domain': domain})
        q = {'proposition': "token.location equals 'A hand'", 'observation_domain': 'state_attribute'}
        before = deepcopy(q)
        qualify_state_actors(q, {})
        self.assertEqual(q, before)

    def test_reported_old_question_id_is_a_hand_not_a_generic_hand(self):
        # Reconstruct the reported 48-sample/four-crop request schema. This
        # identifies its question, not what the underlying images really show.
        catalog = {f's0:f{i:03}': {'kind': 'sample'} for i in range(48)}
        for boundary in ('first', 'last'):
            catalog[f's0:{boundary}'] = {'kind': 'boundary'}
            for i in range(4):
                catalog[f's0:{boundary}:crop{i}'] = {'kind': 'crop'}
        citation = citation_contract({'shot_index': 0, 'phase': 'pre'}, {}, catalog)
        citation['refs_when_unknown'] = '[] or allowed IDs explaining uncertainty'
        q = {'proposition': "token.location equals 'A hand'", 'phase': 'pre',
             'shot_index': 0, 'observation_domain': 'state_attribute',
             'citation_contract': citation, 'attribute': 'token.location'}
        self.assertEqual('o_' + stable_hash(q)[:16], 'o_4f7829ec893d7eb9')
        qualify_state_actors(q, self.actors)
        self.assertNotEqual('o_' + stable_hash(q)[:16], 'o_4f7829ec893d7eb9')
        self.assertIn('ACTOR[A]', q['proposition'])
        self.assertEqual(set(q['actor_references']), {'A'})

    def test_all_hypotheses_qualified_equally_target_private_and_sources_immutable(self):
        fx = isolation_fixture.ObservationIsolationTests(); fx.setUp()
        fx.public['metadata']['h3_global_constraints'] = SOURCE
        shot = fx.public['metadata']['story_contract']['shots'][0]
        shot['preconditions']['token.location'] = 'A hand'
        shot['postconditions']['token.location'] = 'B hand'
        original = deepcopy(fx.public)
        plans = {v: compile_plan(fx.rules, fx.public, SPANS, fx.manifest, protocol=v)
                 for v in (LEGACY_VERSION, BOUNDARY_VERSION, VERSION)}
        old = plans[BOUNDARY_VERSION]['isolated_observations']
        self.assertEqual(old, plans[LEGACY_VERSION]['isolated_observations'])
        self.assertTrue(any("'A hand'" in q['proposition'] for q in old['questions'].values()))
        plan = plans[VERSION]
        packet = plan['isolated_observations']
        new = packet['questions']
        for phase in ('pre', 'post'):
            props = {q['proposition'] for q in new.values() if q['phase'] == phase}
            self.assertIn("token.location equals 'ACTOR[A] hand'", props)
            self.assertIn("token.location equals 'ACTOR[B] hand'", props)
        self.assertEqual(new[packet['target_map'][POST_FACT]]['proposition'],
                         "token.location equals 'ACTOR[B] hand'")
        self.assertEqual(plan['sources'][POST_FACT]['proposition'], "token.location equals 'B hand'")
        self.assertNotIn('target_map', wire_contract(plan))
        self.assertEqual(fx.public, original)
        # Other supported hypotheses never fill the actual unknown target.
        fx.plan = plan
        raw = isolation_fixture.response(plan)
        raw['facts'][packet['target_map'][POST_FACT]].update(basis='occluded', evidence_refs=[])
        self.assertIsNone(fx.parsed(raw)['story.s2.post.token.location']['score'])

    def test_versioned_prompts_preserve_replay_and_do_not_claim_semantic_validation(self):
        self.assertEqual(set(SYSTEMS), {LEGACY_VERSION, BOUNDARY_VERSION, ACTOR_VERSION, VERSION})
        self.assertNotIn('ACTOR[label]', SYSTEMS[BOUNDARY_VERSION])
        self.assertIn('ACTOR[label]', SYSTEMS[VERSION])
        self.assertIn('shared or ambiguous', SYSTEMS[VERSION])


if __name__ == '__main__':
    unittest.main()
