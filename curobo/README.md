# PiPER cuRobo on Jetson AGX Thor

이 디렉토리는 **cuRobo 모션 플래너**를 관리합니다. 물리 시뮬레이션과 GUI는
기존 `../mujoco` 프로젝트가 담당합니다. 즉, cuRobo가 충돌 없는 6축 관절
trajectory를 만들고 MuJoCo가 그 trajectory를 실행합니다.

현재 환경은 다음 조합으로 검증되었습니다.

- NVIDIA Jetson AGX Thor (ARM64)
- CUDA 13.0 / PyTorch 2.10.0
- 공식 cuRobo `main`, commit `8e734f3ced1df898990bcd92de40abce475907db`
- Docker image `piper-curobo-thor:latest`
- MuJoCo 3.11.0

## 디렉토리 구조

```text
curobo/
├── bridge/                 # cuRobo trajectory를 MuJoCo에서 재생
├── docker/                 # Thor용 이미지 정의
├── outputs/                # 생성된 trajectory JSON
├── robots/piper/
│   ├── piper_arm.urdf      # MuJoCo Menagerie와 맞춘 6축 기구학
│   ├── piper.yml           # 자동 생성된 충돌구/self-collision 설정
│   └── meshes -> ...       # 기존 PiPER 메시를 공유하는 심볼릭 링크
├── scripts/                # 빌드, 점검, 계획, 재생 명령
├── vendor/curobo/          # 공식 NVIDIA cuRobo 저장소
└── worlds/                 # cuRobo 충돌 환경
```

`models`에 파일을 넣으면 자동으로 시뮬레이터에 등장하는 방식은 아닙니다.
cuRobo에서는 robot YAML과 world YAML을 계획 코드가 명시적으로 읽고,
MuJoCo에서는 scene XML이 로봇과 물체를 명시적으로 불러옵니다.

## 바로 실행하기

프로젝트로 이동합니다.

```bash
cd /home/airlab/.keys/workspace/sim/curobo
```

GPU와 FK를 점검합니다.

```bash
bash scripts/in_container.sh python scripts/check_gpu.py
bash scripts/in_container.sh python scripts/test_piper_fk.py
```

기본 예제는 현재 TCP 자세에서 위로 5 cm 이동하는 경로를 만듭니다.

```bash
bash scripts/plan.sh
```

생성 결과는 `outputs/piper_trajectory.json`입니다. 이를 기존 MuJoCo GR00T
테이블 장면에서 재생합니다.

```bash
bash scripts/play_mujoco.sh
```

계획이 끝나는 즉시 GUI에서 바로 재생하려면 한 명령으로 실행할 수 있습니다.

```bash
bash scripts/plan_and_play.sh --default --hold 10
```

목표 좌표까지 한 번에 지정할 수도 있습니다.

```bash
bash scripts/plan_and_play.sh 0.45 0.00 0.35 --hold 10
```

빨간 블록 윗면 3 cm 위로 접근하는 전용 명령도 준비되어 있습니다.

```bash
bash scripts/go_to_red.sh
```

이 명령은 빨간 블록만 남긴 `piper_red_gate.xml` 장면을 엽니다. 로봇과
빨간 블록 사이에는 네 개의 박스로 만든 세워진 ㅁ자 게이트가 있으며,
cuRobo는 동일한 `worlds/piper_red_gate.yml` 충돌 환경을 사용해 게이트 구멍을
통과하는 경로를 찾습니다. 결과는 `outputs/red_approach_trajectory.json`에
저장된 후 곧바로 MuJoCo GUI에서 재생됩니다.

## 빨간 블록 드래그 + 실시간 재계획

GUI를 계속 켜둔 채 빨간 블록을 옮기고 cuRobo가 다시 계획하게 하려면 다음
한 명령을 실행합니다.

```bash
bash scripts/live_red.sh
```

첫 실행은 cuRobo와 CUDA graph warmup 때문에 약 1분이 걸릴 수 있습니다.
`Planner is ready`가 출력되면 MuJoCo GUI가 열립니다. 그 이후 재계획은 같은
플래너를 재사용하므로 새 컨테이너를 매번 시작하지 않습니다.

GUI 조작법은 다음과 같습니다.

1. 빨간 블록을 더블클릭해 선택합니다.
2. `Ctrl + 마우스 오른쪽 드래그`로 화면의 수직 평면에서 옮깁니다.
3. `Ctrl + Shift + 마우스 오른쪽 드래그`로 수평 평면에서 옮깁니다.
4. 마우스를 놓고 약 0.6초 기다리면 새 위치로 자동 계획하고 로봇이 움직입니다.

반투명 초록 구는 빨간 블록 윗면 3 cm 위의 현재 TCP 목표를 표시합니다.
빨간 블록을 다시 움직이면 실행 중인 궤적을 중단하고 현재 관절 자세에서 새
경로를 계산합니다. GUI를 닫으면 이 명령이 시작한 cuRobo 서버도 함께 종료됩니다.

## 엔드포인트 XYZ만 지정해서 움직이기

물체의 질량·마찰·접촉을 목표로 사용하지 않고, TCP(두 손가락 사이 중심)의
위치만 지정하려면 다음 명령을 실행합니다.

```bash
bash scripts/live_endpoint.sh
```

초록 구는 질량과 충돌이 없는 목표 마커입니다. 빨간 물체는 이 장면에 없으며,
cuRobo는 목표의 회전 자세를 무시하고 초록 구의 MuJoCo world XYZ만 사용합니다.
테이블과 ㅁ자 게이트는 실제 충돌 장애물로 계속 반영됩니다. 초록 구를
더블클릭한 뒤 `Ctrl + 오른쪽 드래그` 또는
`Ctrl + Shift + 오른쪽 드래그`로 옮기고 놓으면, 현재 팔 자세에서 자동으로
새 궤적을 계산해 같은 GUI에서 재생합니다.

실행 중 `O` 키를 누르면 ㅁ자 게이트를 켜거나 끌 수 있습니다. 화면 표시,
MuJoCo 충돌, cuRobo 충돌 월드가 함께 전환되고 현재 목표점으로 자동
재계획합니다. 테이블은 안전을 위해 항상 충돌 환경에 남습니다.

반투명 하늘색 점들은 6축 관절 범위에서 샘플링한 충돌 없는 구성의 TCP 위치,
즉 6차원 C-space를 3차원 작업공간으로 투영한 결과입니다. `C` 키로 점군을
숨기거나 다시 표시할 수 있습니다. `O` 키로 게이트를 토글하면 현재 충돌
환경에 맞춰 점군도 다시 계산됩니다. 이 점군은 가능한 위치의 표본이며 각
점까지 항상 연결된 경로가 존재한다는 보장은 아니므로, 실제 이동 가능 여부는
cuRobo의 궤적 계획 결과가 최종 판단합니다.

PiPER 어깨를 중심으로 한 큰 반투명 파란 구는 표본에서 계산한 최대 도달 반경
경계입니다. `P` 키로 이 구만 켜거나 끌 수 있습니다. 이 구 전체가 충돌 없이
도달 가능하다는 뜻은 아니며, 정확한 가능 영역은 구 내부의 하늘색 점과 실제
cuRobo 계획 결과를 사용해야 합니다.

실행 중 `+` 또는 `=` 키를 누르면 테이블 반대편(`x=0.66 m`)에 두 번째 PiPER가
180도 돌아선 상태로 나타나 기본 팔과 서로 마주봅니다. `-` 키를 누르면 다시
제거됩니다. MuJoCo는 실행 중 모델 구조를 바꿀 수 없으므로 두 번째 팔을 씬에
미리 비활성 상태로 포함하고 표시와 충돌을 토글합니다. 추가된 동안에는 해당
팔의 충돌구도 cuRobo 월드에 들어갑니다. 큰 초록 구 하나가 두 팔의 공통
목표이며 이것만 드래그합니다. 두 팔이 부딪히지 않도록 기본 팔의 실제 목표는
공통점보다 x축으로 6 cm 왼쪽인 작은 청록색 구, 두 번째 팔의 목표는 6 cm
오른쪽인 작은 자홍색 구로 자동 계산됩니다. 한 번의 공통 목표 변경으로 두
cuRobo 궤적을 모두 생성합니다. 먼저 기본 팔의 경로만 계산해서 저장하고 실제로
움직이지는 않습니다. 그 경로의 예상 최종 자세를 충돌구 장애물로 놓고 두 번째
팔의 경로를 계산합니다. 이후 MuJoCo에서 두 경로를 조합해 팔 사이 충돌을
검사하고 두 팔을 함께 실행합니다. 동시 시작 경로에 충돌이 있으면 두 번째 팔의
시작을 0.1초씩 늦춰 가장 빠른 안전 시차를 자동 선택합니다. `/` 키는 두 번째
팔의 실행을 현재 자세에서 정지하거나 다시 활성화합니다.

GUI를 열 때 초기 목표 좌표(m)를 직접 지정할 수도 있습니다.

```bash
bash scripts/live_endpoint.sh --target 0.42 0.08 0.32
```

두 팔 사이의 목표 간격은 기본 12 cm입니다. 팔 하나당 중심점으로부터 떨어질
거리를 변경하려면 다음처럼 지정합니다.

```bash
bash scripts/live_endpoint.sh \
  --target 0.42 0.08 0.32 \
  --dual-goal-offset 0.06
```

처음부터 게이트 없이 시작하려면 다음 옵션을 사용합니다.

```bash
bash scripts/live_endpoint.sh --no-obstacles --target 0.42 0.08 0.32
```

이 좌표는 **MuJoCo world 기준**입니다. 초록 구를 장애물 내부나 로봇의 도달
범위 밖으로 옮기면 충돌 없는 경로가 없으므로 계획 실패 메시지가 나오는 것이
정상입니다. GUI를 닫으면 이 명령이 시작한 cuRobo 서버도 함께 종료됩니다.

GUI 없이 연결만 검사하려면 다음과 같이 실행합니다.

```bash
bash scripts/play_mujoco.sh --headless --hold 0.2
```

이 재생기는 별도 서버를 띄우지 않습니다. GUI를 닫으면 해당 터미널의
프로세스도 끝납니다.

## 원하는 위치로 계획하기

목표 위치 단위는 m이고 **PiPER 베이스 기준 좌표**입니다.

```bash
bash scripts/plan.sh --goal-position 0.45 0.00 0.35
bash scripts/play_mujoco.sh
```

자세도 지정할 경우 quaternion 순서는 `qw qx qy qz`입니다.

```bash
bash scripts/plan.sh \
  --goal-position 0.45 0.00 0.35 \
  --goal-quaternion 1 0 0 0
```

시작 관절각도 rad 단위로 지정할 수 있습니다.

```bash
bash scripts/plan.sh --start 0 1.57 -1.35 0 0 0
```

`worlds/piper_tabletop.yml`의 z=0은 로봇 장착면입니다. 대응하는 MuJoCo
장면에서는 로봇 베이스가 world z=0.20 m에 있으므로 좌표 변환은 다음과
같습니다.

```text
MuJoCo world position = cuRobo base position + [0, 0, 0.20]
```

충돌 환경을 잠시 빼고 IK/경로 문제만 진단할 때는 `--no-world`를 사용합니다.

```bash
bash scripts/plan.sh --no-world --goal-position 0.45 0.00 0.35
```

## 환경을 다시 만드는 명령

이미 이미지와 PiPER 설정은 만들어져 있으므로 평소에는 실행할 필요가 없습니다.

```bash
bash scripts/build_image.sh
bash scripts/build_piper_config.sh
```

두 번째 명령은 PiPER STL로 충돌구와 자기충돌 행렬을 다시 최적화하므로
Thor에서 약 5분 걸릴 수 있습니다.

## 현재 범위와 주의점

- cuRobo 계획 자유도는 `joint1`부터 `joint6`까지입니다.
- TCP는 두 손가락 사이의 `gripper_center`입니다.
- 그리퍼 개폐는 경로 계획과 분리되어 있으며 MuJoCo 재생 시 기본 40 mm로
  유지됩니다. `--gripper-opening 0.02`처럼 변경할 수 있습니다.
- 현재 cuRobo world에는 테이블과 흰색 플레이트가 포함됩니다. 움직이는 색상
  블록을 장애물로 피하려면 매 계획 시점의 블록 pose를 world에 갱신해야 합니다.
- 실제 로봇으로 보내기 전에는 별도의 관절 부호/영점 확인, 속도 제한, 비상정지,
  실제 충돌 검증이 반드시 필요합니다. 현재 브리지는 시뮬레이션 전용입니다.
