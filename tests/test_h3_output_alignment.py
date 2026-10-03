import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from evovideo_skill.h3_api import H3GenerationTool
from evovideo_skill.h3_local import build_local_request
from evovideo_skill.h3_media import H3AVConcatTool, _command, _probe
from evovideo_skill.h3_output_alignment import align_story_output, sha256
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.planning import Planner
from evovideo_skill.tools import ToolExecutionContext
from evovideo_skill.video_processing import VideoProcessor


class OutputAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def video(self, duration, name='raw'):
        path=self.root/f'{name}.mp4'
        _command(['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i',
                  f'testsrc2=size=64x48:rate=24:duration={duration}',
                  '-f','lavfi','-i',f'sine=sample_rate=32000:duration={duration}',
                  '-c:v','libx264','-c:a','aac','-t',str(duration),str(path)])
        return path

    def test_three_overlong_shots_align_to_18_without_editing_original(self):
        raw=self.video(158/24)
        before=sha256(raw)
        clip,record=align_story_output(raw,6,self.root)
        self.assertEqual(sha256(raw),before)
        self.assertAlmostEqual(record['source_video_seconds'],158/24,places=3)
        self.assertAlmostEqual(_probe(clip,'video')['duration_seconds'],6,places=3)
        self.assertTrue(_probe(clip,'video')['has_audio'])
        task=VideoTask.from_dict({'task_id':'story','prompt':'Three ordered shots','duration_seconds':18})
        inputs={f's{i}':VideoArtifact(f'a{i}',task.task_id,task.prompt,task.mode,[],[],
                      {'artifact_type':'video','local_video_path':str(clip),'h3_output_alignment':record}) for i in range(3)}
        result=H3AVConcatTool(self.root/'concat').run_with_context(task,Planner().plan(task,[]),
                  ToolExecutionContext('join',{'source_nodes':list(inputs)},inputs))
        self.assertAlmostEqual(_probe(Path(result.metadata['local_video_path']),'video')['duration_seconds'],18,places=3)
        self.assertEqual(len(result.metadata['concat_sources']),3)
        self.assertTrue(all('h3_output_alignment' in v for v in result.metadata['concat_sources']))

    def test_reject_short_or_large_overrun_and_reuse_verified_result(self):
        for duration in (5.8,6.9):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError,'only trims'):
                align_story_output(self.video(duration,str(duration)),6,self.root)
        raw=self.video(6.5)
        output,record=align_story_output(raw,6,self.root)
        stamp=output.stat().st_mtime_ns
        again,second=align_story_output(raw,6,self.root)
        self.assertEqual(output,again);self.assertEqual(stamp,again.stat().st_mtime_ns)
        self.assertEqual(record,second)
        output.write_bytes(output.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            align_story_output(raw,6,self.root)

    def test_exact_video_ignores_aac_container_tail(self):
        source=self.video(6)
        aligned,record=align_story_output(source,6,self.root)
        self.assertEqual(aligned,source)
        self.assertEqual(record['operation'],'none')

    def test_all_native_story_modes_normalize_before_sampling_and_direct_concat(self):
        raw=self.video(158/24)
        image=self.root/'ref.png'
        _command(['ffmpeg','-v','error','-nostdin','-y','-i',str(raw),'-frames:v','1',str(image)])
        ref={'id':'hero','kind':'image','role':'reference_image','uri':str(image)}
        task=VideoTask.from_dict({'task_id':'story','prompt':'Three shots','duration_seconds':18,
             'metadata':{'story_contract':{'version':1},'generation_seed':42,'h3_references':[ref],
                         'h3_shots':[{'prompt':f'Shot {i}','duration_seconds':6} for i in range(3)]}})
        calls=[]
        def generate(payload, identity):
            calls.append(payload)
            return raw.stem,raw.as_uri(),{'request_hash':'fixture','seed':42}
        client=SimpleNamespace(root=self.root,provider_name='local-h3',provider_seed_control=True,
             config=SimpleNamespace(resolution='768P',ratio='16:9'),build_request=build_local_request,
             generate=generate,processor=VideoProcessor(self.root,3))
        plan=Planner().plan(task,[])
        for mode in ('t2va','ref2va','fl2va'):
            reference={**ref,'role':'first_frame' if mode=='fl2va' else 'reference_image'}
            upstream=VideoArtifact('refs',task.task_id,task.prompt,task.mode,[],[],
                                  {'artifact_type':'h3_reference_set','h3_references':[reference]})
            context=ToolExecutionContext('candidate',{'shot_index':0},{'refs':upstream} if mode!='t2va' else {})
            result=H3GenerationTool(client,mode).run_with_context(task,plan,context)
            self.assertEqual(result.metadata['h3_output_alignment']['requested_seconds'],6)
            self.assertAlmostEqual(_probe(Path(result.metadata['local_video_path']),'video')['duration_seconds'],6,places=3)
            self.assertIn('aligned-story-',result.metadata['sampled_frame_paths'][0])
        baseline=H3GenerationTool(client,'direct').run(task,plan)
        self.assertEqual(len(baseline.metadata['h3_output_alignments']),3)
        self.assertAlmostEqual(_probe(Path(baseline.metadata['local_video_path']),'video')['duration_seconds'],18,places=3)
        self.assertEqual(len(calls),6)


if __name__=='__main__':unittest.main()
