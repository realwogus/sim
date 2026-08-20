# AgileX PiPER MuJoCo

Jetson AGX Thor에서 AgileX PiPER를 시뮬레이션하고 강화학습 환경으로 확장하기 위한
독립 프로젝트다. Isaac Sim 프로젝트와 의존성을 공유하지 않는다.

## 포함된 구성

- Google DeepMind MuJoCo Menagerie의 공식 PiPER MJCF
- PiPER 모델 로딩 및 headless smoke test
- 관절 position-control 예제
- Gymnasium 호환 end-effector reaching 환경
- GUI viewer 및 RGB offscreen rendering 예제
- Stable-Baselines3 PPO 학습 진입점(선택 의존성)

공식 모델은 `models/mujoco_menagerie/agilex_piper/`에 sparse clone되어 있다. 이
디렉터리는 upstream 비교가 가능하도록 직접 수정하지 않는다.

## 시작하기

```bash
cd /home/airlab/.keys/workspace/sim/mujoco
source .venv/bin/activate

python scripts/smoke_test.py
python scripts/random_control.py --seconds 5
python scripts/render_frame.py --output outputs/piper.png
python -m pytest
```

현재 X11 데스크톱에서 viewer를 열려면 다음을 실행한다.

```bash
python scripts/view_piper.py
```

두 PiPER가 서로 마주보는 씬은 다음처럼 실행한다.

```bash
python scripts/view_scene.py
```

학습 영상의 검은 테이블, 흰 접시, 여러 색 블록과 3개 정책 카메라를 재현한
GR00T 태스크 씬은 다음처럼 확인한다.

```bash
python scripts/view_gr00t_task.py
MUJOCO_GL=egl python scripts/render_gr00t_cameras.py
```

파인튜닝된 GR00T 체크포인트를 MuJoCo에 연결할 때는 두 터미널을 사용한다.

```bash
# 터미널 1: GPU 추론 서버
bash scripts/start_gr00t_server.sh

# 터미널 2: MuJoCo 클라이언트
python -m pip install -e '.[gr00t]'
python scripts/run_gr00t_mujoco.py --task red
```

20,000-step 체크포인트는 전용 스크립트로 실행한다.

```bash
# 터미널 1
bash scripts/start_gr00t_checkpoint_20000.sh

# 터미널 2
python scripts/run_gr00t_mujoco.py --task red
```

일반 서버 스크립트에 체크포인트 경로를 직접 넘겨도 동일하다.

```bash
bash scripts/start_gr00t_server.sh gr00t_outputs/checkpoint-20000
```

정책 입력은 학습과 동일하게 `wrist`, `right`, `left` RGB 영상과 6개 관절,
전체 그리퍼 개방 폭, 영어 태스크 문장을 사용한다. 카메라 외부 파라미터와 블록
치수는 영상으로 추정한 값이므로 실제 촬영 캘리브레이션을 확보하면 교체해야 한다.

독립 viewer에서 직접 열려면 다음 명령을 사용한다.

```bash
python -m mujoco.viewer --mjcf=models/scenes/dual_piper.xml
```

`models/scenes/dual_piper.xml`은 Menagerie의 동일한 `piper.xml`을 두 번
attach한다. 이름 충돌을 피하기 위해 각각 `left_`, `right_` prefix를 사용하며,
base 위치는 `(0, -0.45, 0)`과 `(0, 0.45, 0)`이다.

PPO 학습 의존성은 기본 검증 후 선택적으로 설치한다.

```bash
python -m pip install -e '.[train]'
python scripts/train.py --steps 100000
```

기본 PPO MLP는 CPU에서 학습한다. 이 규모에서는 GPU 전송 비용 때문에 CPU가 더
효율적이다. 설치된 CUDA PyTorch와 Thor GPU는 cuRobo 및 향후 GPU 병렬 물리 환경에
사용한다.

## 환경 정의

`PiperReachEnv`의 action은 7차원 `[-1, 1]` 값이다. 앞의 6개 값은 각 암 관절의
position target 증분, 마지막 값은 그리퍼 position target 증분으로 변환된다.

observation은 다음 값을 연결한다.

```text
qpos(8) + qvel(8) + end-effector position(3) + target position(3)
```

현재 end-effector 기준점은 `link6` body origin이다. 실제 grasp 학습 단계에서는
프로젝트용 MJCF에 TCP site를 추가하고 이를 기준점으로 바꾼다.

## 다음 확장

1. TCP site와 grasp site 추가
2. table/object가 포함된 pick-and-place scene 작성
3. actuator 및 마찰 파라미터 실기 캘리브레이션
4. dual-arm scene과 두 PiPER 간 collision 설정
5. cuRobo PiPER 설정 및 collision sphere 연결
