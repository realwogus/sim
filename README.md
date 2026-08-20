# PiPER Simulation Workspace

AgileX PiPER의 MuJoCo 물리 시뮬레이션과 cuRobo GPU 모션 플래닝을 함께
관리하는 워크스페이스다. cuRobo가 충돌을 피하는 관절 궤적을 생성하고,
MuJoCo가 로봇·물체의 물리 상태를 계산하며 GUI에서 궤적을 재생한다.

현재 구성은 NVIDIA Jetson AGX Thor, ARM64, CUDA 13.0 환경에서 검증했다.

## 디렉토리 구조

```text
sim/
├── mujoco/                         # 물리 시뮬레이션, 장면, GUI, 학습 환경
├── curobo/                         # GPU 궤적 계획과 MuJoCo 연동 코드
│   └── vendor/curobo/              # NVIDIA cuRobo 서브모듈
└── third_party/
    └── piper_isaac_sim/            # PiPER 메시 자산 서브모듈
```

MuJoCo Menagerie도 `mujoco/models/mujoco_menagerie/`에 서브모듈로 고정되어
있다. 로봇과 장면 파일은 `mujoco/models/`, cuRobo 로봇 설정은
`curobo/robots/`, 장애물 월드는 `curobo/worlds/`에서 관리한다.

## 저장소 받기

세 외부 저장소를 서브모듈로 사용하므로 반드시 함께 받아야 한다.

```bash
git clone --recurse-submodules https://github.com/realwogus/sim.git
cd sim
```

이미 일반 `git clone`으로 받은 경우에는 다음 명령을 추가로 실행한다.

```bash
git submodule update --init --recursive
```

서브모듈이 정상적으로 준비됐는지 확인한다.

```bash
git submodule status
test -d curobo/vendor/curobo/curobo
test -d mujoco/models/mujoco_menagerie/agilex_piper
test -d third_party/piper_isaac_sim/piper_description/meshes
```

## Git에 포함되지 않아 별도로 준비할 항목

다음 항목은 크기가 크거나 장비마다 달라 저장소에 포함하지 않는다.

| 항목 | 필요한 기능 | 준비 방법 |
|---|---|---|
| Python 가상환경 `mujoco/.venv` | 모든 MuJoCo 실행 | 아래의 MuJoCo 설치 명령 사용 |
| NVIDIA 드라이버, CUDA, Docker, NVIDIA Container Toolkit | cuRobo GPU 계획 | 호스트 장비에 별도 설치 |
| `gr00t-thor:latest` Docker 이미지 | 현재 Thor용 cuRobo 이미지의 기반 | 기존 GR00T 환경에서 별도 빌드 |
| `piper-curobo-thor:latest` 이미지 | cuRobo 실행 | `bash curobo/scripts/build_image.sh` |
| Isaac-GR00T 소스와 학습 체크포인트 | GR00T 정책 추론에만 필요 | 선택 설치, 아래 GR00T 절 참고 |
| Hugging Face 모델 캐시 | GR00T 추론에만 필요 | 모델 다운로드 또는 기존 캐시 지정 |
| `outputs/`, 로그, 학습 결과 | 실행 중 생성 | 다시 실행하면 생성됨 |

즉, MuJoCo만 사용할 때는 GPU Docker 환경이나 GR00T 체크포인트가 필요하지
않다. cuRobo를 사용할 때는 NVIDIA GPU Docker 환경이 필요하고, GR00T는 정책
추론을 할 때만 추가로 필요하다.

## 1. MuJoCo 환경 설치

Python 3.10 이상이 필요하다. Ubuntu에서 가상환경 기능이 없다면 먼저
`python3-venv`를 설치한다.

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libglfw3 libxinerama1 libxcursor1 libxi6

cd /path/to/sim/mujoco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

강화학습 또는 GR00T 클라이언트도 사용할 경우 선택 의존성을 추가한다.

```bash
# Stable-Baselines3 PPO 학습
python -m pip install -e '.[train]'

# GR00T 추론 서버와 통신하는 MuJoCo 클라이언트
python -m pip install -e '.[gr00t]'
```

설치와 모델 로딩을 확인한다.

```bash
python scripts/smoke_test.py
python -m pytest
```

GUI가 있는 Linux 데스크톱에서 장면을 연다.

```bash
python scripts/view_scene.py
```

SSH 접속 환경에서는 X11/Wayland 디스플레이 전달이 별도로 필요하다.

## 2. cuRobo Docker 환경 설치

현재 `curobo/docker/Dockerfile.thor`는 `gr00t-thor:latest` 이미지와 CUDA 13을
기준으로 한다. 다음 명령이 실패하면 먼저 NVIDIA 드라이버, Docker 및 NVIDIA
Container Toolkit을 설치해야 한다.

```bash
nvidia-smi
docker run --rm --runtime nvidia --gpus all ubuntu:24.04 nvidia-smi
docker image inspect gr00t-thor:latest
```

기반 이미지가 준비된 후 cuRobo 이미지를 만들고 검사한다.

```bash
cd /path/to/sim/curobo
bash scripts/build_image.sh
bash scripts/in_container.sh python scripts/check_gpu.py
bash scripts/in_container.sh python scripts/test_piper_fk.py
```

`piper-curobo-thor:latest`가 이미 있는 장비에서는 이미지 빌드를 생략할 수 있다.
x86_64 PC나 CUDA 버전이 다른 장비에서는 해당 환경에 맞는 기반 이미지와 cuRobo
CUDA extra를 사용하도록 Dockerfile을 조정해야 한다.

## 3. MuJoCo와 cuRobo 통합 GUI 실행

MuJoCo 가상환경과 cuRobo Docker 이미지가 모두 준비된 뒤 실행한다.

```bash
cd /path/to/sim/curobo
bash scripts/live_endpoint.sh
```

스크립트는 `sim` 디렉토리를 컨테이너의 `/workspace`로 마운트한다. 상세 조작법과
목표 좌표, 장애물 및 두 번째 로봇 키 설정은 `curobo/README.md`에 정리되어 있다.

## 4. GR00T 추론 환경(선택)

GR00T는 이 저장소나 서브모듈에 포함되지 않는다. 사용할 때는 다음을 별도로
준비해야 한다.

- NVIDIA Isaac-GR00T 저장소
- 학습된 체크포인트 디렉토리
- `gr00t-thor:latest` 이미지
- 모델을 받은 Hugging Face 캐시

현재 장비처럼 GR00T 소스와 체크포인트가 `sim` 밖에 있다면 환경 변수로 실제
절대 경로를 지정한다.

```bash
cd /path/to/sim/mujoco
source .venv/bin/activate

export GR00T_REPO=/absolute/path/to/Isaac-GR00T
export GR00T_CHECKPOINT=/absolute/path/to/gr00t_outputs/checkpoint-20000
export GR00T_PERSISTENT_HOME=/absolute/path/to/persistent/home

# 터미널 1: GPU 추론 서버
bash scripts/start_gr00t_server.sh

# 터미널 2: MuJoCo 클라이언트
python scripts/run_gr00t_mujoco.py --task red
```

체크포인트와 Hugging Face 캐시는 용량이 크므로 Git에 커밋하지 않는다.

## 새 장비에서 최소 설치 순서

```text
저장소와 서브모듈 clone
  -> MuJoCo Python 가상환경 설치
  -> MuJoCo smoke test
  -> NVIDIA Docker 환경 확인
  -> cuRobo 이미지 빌드 및 FK 검사
  -> 통합 GUI 실행
  -> 필요할 때만 GR00T 소스와 체크포인트 연결
```
