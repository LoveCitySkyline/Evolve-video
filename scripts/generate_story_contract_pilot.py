#!/usr/bin/env python3
"""Build a reviewable draft story corpus, not a validated benchmark or result."""
import json
from pathlib import Path

# Each tuple is an independently specified scenario: state key, initial value,
# six observable events and their resulting states. No scenario crosses splits.
SCENARIOS = [
('key_escape', 'key.owner', 'Mira', [('Mira shows the red key to Lin','Mira'),('Mira hands the key to Lin and releases it','Lin'),('Lin unlocks the iron door while holding the key','Lin'),('Lin leads Mira through the doorway with the key visible','Lin'),('Lin returns the key to Mira','Mira'),('Mira locks the door behind them with the key','Mira')]),
('paper_crane', 'paper.form', 'flat square', [('A child folds the square diagonally','triangle'),('The child folds it into a small diamond','diamond'),('The child opens one flap into a neck','partly formed crane'),('The child folds a pointed head','crane with folded wings'),('The child spreads both wings','open-winged crane'),('The child sets the same crane on a windowsill','open-winged crane on sill')]),
('lantern_rescue', 'lantern.location', 'table', [('A sailor lifts a brass lantern','sailor hand'),('The sailor carries it down steps','bottom of steps'),('The sailor places it by a stranded boat','boat edge'),('A passenger lifts the lantern','passenger hand'),('The passenger hangs it on a hook aboard','boat hook'),('The lantern illuminates both people as they depart','boat hook')]),
('train_ticket', 'ticket.holder', 'Iris', [('Iris displays a blue ticket','Iris'),('A gust pulls the ticket from her fingers','air'),('The ticket lands beside a bench','ground'),('A porter picks it up','porter'),('The porter returns it to Iris','Iris'),('Iris presents it to the conductor','conductor')]),
('broken_bridge', 'plank.location', 'shed', [('A worker lifts a wooden plank from the shed','worker hands'),('The worker carries it to a stream','stream bank'),('The worker lays it across two stones','across stream'),('The worker tests the plank with one foot','across stream'),('A cyclist walks across carrying a bicycle','across stream'),('The worker retrieves the plank after the cyclist crosses','worker hands')]),
('paint_restoration', 'sign.appearance', 'dusty', [('A painter wipes dust from an old sign','clean bare sign'),('The painter applies a white base coat','white base coat'),('The painter draws a blue outline','blue outline'),('The painter fills a blue bird inside the outline','blue bird'),('The painter adds a yellow sun beside the bird','blue bird and yellow sun'),('The painter hangs the completed sign above the shop','hung blue bird and yellow sun sign')]),
('parcel_handoff', 'parcel.location', 'courier bag', [('A courier removes a striped parcel from a bag','courier hands'),('The courier places the parcel on a counter','counter'),('A clerk checks its label without moving it','counter'),('The clerk hands the parcel to a customer','customer hands'),('The customer carries it to a bicycle basket','bicycle basket'),('The customer secures the same parcel with a strap','strapped in basket')]),
('plant_revival', 'plant.location', 'dry outdoor pot', [('A gardener lifts a wilted plant and root ball','gardener hands'),('The gardener lowers it into a larger pot','larger pot'),('The gardener adds soil around the root ball','larger pot with soil'),('The gardener waters the soil','watered larger pot'),('The gardener carries the pot into shade','shaded bench'),('The gardener leaves the pot by a window and steps away','shaded bench')]),
('lost_glasses', 'glasses.location', 'desk', [('An archivist places round glasses on a book','book'),('The archivist closes the book with glasses still on top','closed book'),('An assistant moves the book and glasses to a cart','cart'),('The archivist searches the empty desk','cart'),('The assistant points to the glasses on the cart','cart'),('The archivist retrieves and wears the glasses','archivist face')]),
('rope_crossing', 'rope.attachment', 'coiled on ground', [('A climber picks up a coiled rope','climber hand'),('The climber ties one end around a tree','tree'),('The climber throws the loose end across a gap','tree and far bank'),('A partner catches the loose end','tree and partner hand'),('The partner secures that end to a post','tree and post'),('The climber tests the taut rope','tree and post')]),
('bakery_delivery', 'cake.location', 'baking tray', [('A baker lifts a small cake onto a board','cake board'),('The baker coats it with white icing','iced on board'),('The baker adds three strawberries','decorated on board'),('The baker slides the board into a box','open box'),('The baker closes the box and hands it to a courier','courier hands'),('The courier places the box on a waiting table','waiting table')]),
('power_repair', 'lamp.state', 'off with unplugged cable', [('An electrician shows the disconnected plug','off with unplugged cable'),('The electrician inserts the plug into the socket','off with connected cable'),('The electrician toggles the switch but the lamp stays dark','off with connected cable'),('The electrician replaces the bulb','new bulb with connected cable'),('The electrician toggles the switch and the lamp lights','lit'),('The electrician steps back as the lamp stays lit','lit')]),
('museum_return', 'medallion.location', 'display stand', [('A curator lifts a silver medallion with gloves','curator hand'),('The curator places it on a velvet pad','velvet pad'),('The curator rotates the pad to show the reverse','velvet pad reverse visible'),('A conservator brushes dust off the medallion','velvet pad'),('The curator lifts the cleaned medallion','curator hand'),('The curator returns it to the original stand','display stand')]),
('flood_barrier', 'sandbag.location', 'stack', [('A volunteer lifts a yellow sandbag','volunteer hands'),('The volunteer hands it to a second person','second person hands'),('The second person carries it to a leaking gate','gate edge'),('The second person sets it across the leak','across gate leak'),('Water collects behind the bag instead of crossing','across gate leak'),('Both volunteers point to the reduced flow','across gate leak')]),
('stage_reveal', 'curtain.state', 'closed', [('A stagehand shows the closed red curtain','closed'),('The stagehand grasps a pull rope','closed'),('The stagehand pulls and the curtain opens halfway','half open'),('The stagehand pulls again to reveal a piano','fully open'),('A pianist walks to the revealed piano','fully open'),('The pianist bows while the curtain remains open','fully open')]),
('compass_route', 'compass.holder', 'guide', [('A guide holds an open compass beside a trail fork','guide'),('The guide rotates to align its needle','guide'),('The guide passes the compass to a hiker','hiker'),('The hiker uses it to select the left trail','hiker'),('The hiker leads the guide along the left trail','hiker'),('At a marker the hiker returns the compass','guide')]),
('water_transfer', 'jug.contents', 'empty', [('A cook shows an empty clear jug','empty'),('The cook fills it halfway from a tap','half full of water'),('The cook carries it to a table without spilling','half full of water'),('The cook pours some water into a blue cup','quarter full of water'),('The cook pours the rest into a red cup','empty'),('The cook sets the empty jug between both filled cups','empty')]),
('letter_seal', 'letter.state', 'unfolded', [('A writer signs a sheet of cream paper','signed unfolded'),('The writer folds the sheet twice','folded'),('The writer inserts it into a green envelope','inside open envelope'),('The writer closes the envelope flap','closed envelope'),('The writer presses a red wax seal onto the flap','sealed envelope'),('The writer hands the sealed envelope to a messenger','sealed envelope in messenger hand')]),
]

def build():
    tasks = []
    for number, (name, key, initial, beats) in enumerate(SCENARIOS):
        split = ('train', 'validation', 'test')[number // 6]
        shots, contracts, current = [], [], initial
        for i, (event, after) in enumerate(beats):
            shots.append({'prompt': f'{event}. Show the whole action, keep the same people, object and scene layout. Before: {key}={current}. After: {key}={after}.', 'duration_seconds': 6})
            contracts.append({'shot_index': i, 'preconditions': {key: current}, 'postconditions': {key: after},
                              'events': [{'id': f'{name}_{i}', 'description': event}], 'threshold': .9})
            current = after
        prompt = 'Create a coherent six-shot narrative. ' + '. Then '.join(e for e, _ in beats) + '. Preserve character and prop identity between shots.'
        tasks.append({'task_id': 'story-'+name, 'prompt': prompt, 'duration_seconds': 36, 'mode': 'generation',
            'metadata': {'task_family': 'narrative_continuity', 'split': split, 'scenario_group': name,
                'benchmark_status': 'draft_requires_human_and_asset_review', 'h3_shots': shots,
                'h3_global_constraints': prompt,
                'evaluation': {'identity_across_shots': {'description': 'Characters and props keep their identities across all six shots.', 'threshold': .9, 'mandatory': True, 'weight': 1, 'aggregation': 'minimum_over_segments'},
                               'shot_and_action_coverage': {'description': 'All six declared shots and actions appear in their declared order.', 'threshold': .9, 'mandatory': True, 'weight': 1}},
                'story_contract': {'version': 1, 'initial_state': {key: initial}, 'final_state': {key: current}, 'shots': contracts}}})
    return {'name': 'Story Contract Pilot 18', 'version': 'draft-v1',
            'description': 'Authored contract fixtures for integration and pilot studies; not a validated public benchmark.', 'tasks': tasks}

if __name__ == '__main__':
    destination = Path('benchmarks/story_contract_pilot18.json')
    destination.write_text(json.dumps(build(), ensure_ascii=False, indent=2)+'\n')
    print(destination)
