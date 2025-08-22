import taichi as ti
import numpy as np
import torch
import pandas as pd
import os
from PIL import Image
import io
import psutil
import gc

# Taichi 초기화 - 10GB 메모리 할당 (150만개 입자 지원)
ti.init(arch=ti.gpu)

DIM = 3
MAX_PARTICLES = 200000  # 10GB 메모리 할당을 위해 150만개로 설정
G = 3.568*10**-33
# 중력 상수를 증가시켜 위성 형성 촉진
EPS = 0.05
DT = 0.05#0.0006

PARTICLE_RADIUS_MANTLE = 0.02
PARTICLE_RADIUS_CORE = PARTICLE_RADIUS_MANTLE * 4.0

pos = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
vel = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
acc = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
mass = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

@ti.kernel
def compute_gravity(n: ti.i32):
    for i in range(n):
        acc[i] = ti.Vector.zero(ti.f32, DIM)
        for j in range(n):
            if i != j:
                r = pos[j] - pos[i]
                dist_sqr = r.dot(r) + EPS**2
                acc[i] += G * mass[j] * r.normalized() / dist_sqr

@ti.kernel
def update(n: ti.i32, dt: ti.f32):
    for i in range(n):
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt

@ti.kernel
def correct_overlap(n: ti.i32, r_core: ti.f32, r_mantle: ti.f32):
    for i in range(n):
        for j in range(i + 1, n):
            r = pos[j] - pos[i]
            dist = r.norm()
            min_dist = PARTICLE_RADIUS_MANTLE * 2.0
            if dist < min_dist and dist > 1e-5:
                delta = r.normalized() * (min_dist - dist) * 0.5
                pos[i] -= delta
                pos[j] += delta

@ti.kernel
def apply_cohesion(n: ti.i32, stiffness: ti.f32, radius_limit: ti.f32):
    for i in range(n):
        center = ti.Vector.zero(ti.f32, DIM)
        count = 0
        for j in range(n):
            if i != j:
                r = pos[j] - pos[i]
                if r.norm() < radius_limit:
                    center += pos[j]
                    count += 1
        if count > 0:
            center /= count
            vel[i] += (center - pos[i]) * stiffness

@ti.kernel
def clamp_escape_particles(n: ti.i32, center: ti.types.vector(3, ti.f32), max_dist: ti.f32):
    for i in range(n):
        r = (pos[i] - center).norm()
        if r > max_dist:
            # 이탈 입자들의 속도를 줄이되, 위성 형성을 위해 완전히 멈추지 않음
            vel[i] *= 0.8
            # 지구 방향으로 약간의 속도 추가 (위성 궤도 형성 촉진)
            if r < max_dist * 1.5:  # 너무 멀리 가지 않은 입자들
                toward_earth = (center - pos[i]).normalized()
                vel[i] += toward_earth * 0.1

@ti.kernel
def reinforce_core(n: ti.i32, center: ti.types.vector(3, ti.f32), strength: ti.f32, core_radius: ti.f32):
    for i in range(n):
        r = pos[i] - center
        dist = r.norm()
        if dist < core_radius:
            acc[i] += -r.normalized() * strength * (core_radius - dist)

@ti.kernel
def guide_orbital_particles(n: ti.i32, center: ti.types.vector(3, ti.f32), min_r: ti.f32, G_central: ti.f32):
    for i in range(n):
        r = pos[i] - center
        dist = r.norm()
        if dist > min_r:
            dir = ti.Vector([-r[1], r[0], 0.0]).normalized()
            vel[i] = dir * ti.sqrt(G_central / dist)

def sample_sphere(N, radius, center):
    points = []
    while len(points) < N:
        p = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(p)
        if norm <= 1:
            scaled = (np.random.rand() ** (1/3)) * radius
            points.append(center + p / norm * scaled)
    return np.array(points)

def sample_shell_volume(N, r_min, r_max, center):
    points = []
    while len(points) < N:
        p = np.random.normal(0, 1, 3)
        norm = np.linalg.norm(p)
        if norm > 0:
            unit = p / norm
            r = ((np.random.rand() * (r_max**3 - r_min**3)) + r_min**3) ** (1/3)
            points.append(center + unit * r)
    return np.array(points)

def get_memory_usage():
    """현재 메모리 사용량을 확인하는 함수"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024**3)  # GB 단위로 변환
    return memory_gb

def print_memory_status():
    """메모리 상태를 출력하는 함수"""
    memory_gb = get_memory_usage()
    print(f"현재 메모리 사용량: {memory_gb:.2f} GB")
    
    # 시스템 전체 메모리 정보
    system_memory = psutil.virtual_memory()
    print(f"시스템 전체 메모리: {system_memory.total / (1024**3):.2f} GB")
    print(f"사용 가능한 메모리: {system_memory.available / (1024**3):.2f} GB")
    print(f"메모리 사용률: {system_memory.percent:.1f}%")

def cleanup_memory():
    """메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("메모리 정리 완료")

def initialize_body(center, N_core, N_mantle, R_body, core_fraction, M_total, velocity=np.zeros(3)):
    pos_list, vel_list, mass_list = [], [], []
    R_core = R_body * (core_fraction ** (1/3))
    M_core = (M_total * core_fraction) / N_core
    M_mantle = (M_total * (1 - core_fraction)) / N_mantle
    core_points = sample_sphere(N_core, R_core * 0.7, center)
    mantle_points = sample_shell_volume(N_mantle, R_core * 0.7 + 0.005, R_body, center)
    for p in core_points:
        pos_list.append(p)
        vel_list.append(velocity)
        mass_list.append(M_core)
    for p in mantle_points:
        pos_list.append(p)
        vel_list.append(velocity)
        mass_list.append(M_mantle)
    return pos_list, vel_list, mass_list

class ParticleGenerator(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_particles=500, noise_dim=100):
        super(ParticleGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        self.noise_dim = noise_dim

        self.input_processor = torch.nn.Sequential(
            torch.nn.Linear(input_size * (num_particles), hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3)
        )

        self.noise_processor = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3)
        )

        self.combined_processor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3)
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size // 2, input_size * num_particles),
            torch.nn.Tanh()
        )

    def forward(self, input_sequence, noise):
        batch_size = input_sequence.shape[0]
        last_frame = input_sequence[:, -1, :, :]
        flattened_input = last_frame.view(batch_size, -1)
        input_features = self.input_processor(flattened_input)
        noise_features = self.noise_processor(noise)
        combined_features = torch.cat([input_features, noise_features], dim=1)
        processed_features = self.combined_processor(combined_features)
        output = self.output_layer(processed_features)
        output = output.view(batch_size, self.num_particles, self.input_size)
        return output

_gan_generator_cache = {"path": None, "model": None, "device": None}

def _get_gan_generator(model_path, input_size=4, hidden_size=256, num_particles=500, noise_dim=100):
    try:
        if (_gan_generator_cache["model"] is not None) and (_gan_generator_cache["path"] == model_path):
            return _gan_generator_cache["model"], _gan_generator_cache["device"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ParticleGenerator(input_size=input_size, hidden_size=hidden_size, num_particles=num_particles, noise_dim=noise_dim)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _gan_generator_cache["path"] = model_path
        _gan_generator_cache["model"] = model
        _gan_generator_cache["device"] = device
        return model, device
    except Exception as e:
        print(f"GAN 생성자 로드 실패: {e}. 선형 보간으로 대체합니다.")
        return None, None

def interpolate_with_gan(model_path, positions, velocities, radius=0.01, interpolation_ratio=0.3):
    """
    GAN 생성자를 사용해 시간적 맥락(마지막 프레임 기반)으로 입자 파라미터(x, y, vx, vy)를 예측하고,
    이를 이용해 보간 입자를 생성합니다. z, vz는 기존 분포에서 샘플링해 결합합니다.
    모델 로드 실패 시 기존 선형 보간으로 폴백합니다.
    """
    if len(positions) == 0:
        return []

    num_interpolated = int(len(positions) * interpolation_ratio)
    if num_interpolated <= 0:
        return []

    # 시도: GAN 생성자 로드
    gen, device = _get_gan_generator(model_path)
    if gen is None:
        # 폴백: 기존 선형 보간
        interpolated_particles = []
        for _ in range(num_interpolated):
            idx1 = np.random.randint(0, len(positions))
            idx2 = np.random.randint(0, len(positions))
            alpha = np.random.uniform(0.2, 0.8)
            interpolated_pos = positions[idx1] * alpha + positions[idx2] * (1 - alpha)
            interpolated_vel = velocities[idx1] * alpha + velocities[idx2] * (1 - alpha)
            # z방향 정보를 0으로 설정 (2D 평면에 보간 입자 생성)
            interpolated_pos[2] = 0.0
            interpolated_vel[2] = 0.0
            noise_pos = np.random.normal(0, radius * 0.1, 3)
            noise_vel = np.random.normal(0, 0.01, 3)
            # z방향 노이즈도 0으로 설정
            noise_pos[2] = 0.0
            noise_vel[2] = 0.0
            interpolated_particles.append((interpolated_pos + noise_pos, interpolated_vel + noise_vel))
        return interpolated_particles

    # GAN을 사용한 보간 생성
    generated = []
    remaining = num_interpolated
    gen_particles = getattr(gen, "num_particles", 500)
    noise_dim = getattr(gen, "noise_dim", 100)
    seq_len_minus1 = 4  # 학습 시 윈도우 5, forward는 마지막 프레임만 사용 → 더미로 4회 복제

    # 미리 numpy로 준비
    positions_np = np.asarray(positions, dtype=np.float32)
    velocities_np = np.asarray(velocities, dtype=np.float32)

    while remaining > 0:
        # 조건 프레임: 기존 입자 중 무작위로 gen_particles개 샘플 (x,y,vx,vy)
        if len(positions_np) >= gen_particles:
            sel_idx = np.random.choice(len(positions_np), size=gen_particles, replace=False)
        else:
            sel_idx = np.random.choice(len(positions_np), size=gen_particles, replace=True)
        cond_xy = positions_np[sel_idx, :2]
        cond_vxy = velocities_np[sel_idx, :2]
        last_frame = np.concatenate([cond_xy, cond_vxy], axis=1).astype(np.float32)  # (gen_particles, 4)

        # 입력 시퀀스 구성: 마지막 프레임을 seq_len_minus1번 반복
        input_sequence = np.stack([last_frame for _ in range(seq_len_minus1)], axis=0)  # (seq_len-1, P, 4)
        input_sequence = np.expand_dims(input_sequence, axis=0)  # (1, seq_len-1, P, 4)

        with torch.no_grad():
            inp = torch.from_numpy(input_sequence).to(device)
            noise = torch.randn(1, noise_dim, device=device)
            out = gen(inp, noise).cpu().numpy()[0]  # (P, 4)

        # 생성 결과에서 필요한 개수만 취함
        take = min(remaining, gen_particles)
        gen_xy = out[:take, 0:2]
        gen_vxy = out[:take, 2:4]

        # z, vz는 0으로 설정 (2D 평면에 보간 입자 생성)
        gen_z = np.zeros((take, 1), dtype=np.float32)
        gen_vz = np.zeros((take, 1), dtype=np.float32)

        gen_pos = np.concatenate([gen_xy, gen_z], axis=1)
        gen_vel = np.concatenate([gen_vxy, gen_vz], axis=1)

        # 약간의 노이즈로 퍼짐 보강
        gen_pos += np.random.normal(0, radius * 0.05, gen_pos.shape).astype(np.float32)
        gen_vel += np.random.normal(0, 0.005, gen_vel.shape).astype(np.float32)

        for i in range(take):
            generated.append((gen_pos[i], gen_vel[i]))

        remaining -= take

    return generated

def save_particles_to_csv(positions, velocities, masses, frame_num, csv_filename, basic_particle_count=0, original_csv_filename=None):
    """입자들의 위치, 속도, 질량을 하나의 CSV 파일에 추가"""
    # 각 입자에 대해 프레임 번호와 함께 데이터 생성
    data = []
    
    # 원본 CSV 파일에서 위치 기반 particle_id 매핑 생성
    position_to_id = {}
    if original_csv_filename and frame_num == 0:
        try:
            original_df = pd.read_csv(original_csv_filename)
            frame_0_data = original_df[original_df['frame'] == 0]
            frame_0_data = frame_0_data.sort_values('particle_id')
            
            for idx, row in frame_0_data.iterrows():
                pos_key = (row['x'], row['y'], row['z'])
                position_to_id[pos_key] = row['particle_id']
            
            print(f"원본 CSV에서 위치 기반 particle_id 매핑 생성 완료: {len(position_to_id)}개 위치")
        except Exception as e:
            print(f"원본 CSV 로드 실패: {e}")
            position_to_id = {}
    
    for i in range(len(positions)):
        # 기본 입자인지 보간 입자인지 구분
        particle_type = 'basic' if i < basic_particle_count else 'interpolated'
        
        # 위치 기반 particle_id 결정
        current_pos = (positions[i, 0], positions[i, 1], positions[i, 2])
        
        if current_pos in position_to_id:
            # 원본 CSV에 있는 위치면 원본 particle_id 사용
            particle_id = position_to_id[current_pos]
        else:
            # 원본 CSV에 없는 위치면 순차적 번호 사용
            particle_id = i
        
        data.append({
            'frame': frame_num,
            'particle_id': particle_id,
            'x': positions[i, 0],
            'y': positions[i, 1], 
            'z': positions[i, 2],
            'vx': velocities[i, 0],
            'vy': velocities[i, 1],
            'vz': velocities[i, 2],
            'mass': masses[i],
            'particle_type': particle_type
        })
    
    # 첫 번째 프레임이면 새 파일 생성, 아니면 기존 파일에 추가
    if frame_num == 0:
        with open(csv_filename, 'w') as f:
            f.write("frame,particle_id,x,y,z,vx,vy,vz,mass,particle_type\n")
            for i in range(len(positions)):
                f.write(f"{frame_num},{data[i]['particle_id']},{data[i]['x']:.6f},{data[i]['y']:.6f},{data[i]['z']:.6f},"
                       f"{data[i]['vx']:.6f},{data[i]['vy']:.6f},{data[i]['vz']:.6f},"
                       f"{data[i]['mass']:.6f},{data[i]['particle_type']}\n")
        print(f"새 CSV 파일 {csv_filename}이 생성되었습니다. (입자 수: {len(positions):,}개)")
    else:
        # 기존 파일에 추가 (메모리 효율적)
        with open(csv_filename, 'a') as f:
            for i in range(len(positions)):
                f.write(f"{frame_num},{data[i]['particle_id']},{data[i]['x']:.6f},{data[i]['y']:.6f},{data[i]['z']:.6f},"
                       f"{data[i]['vx']:.6f},{data[i]['vy']:.6f},{data[i]['vz']:.6f},"
                       f"{data[i]['mass']:.6f},{data[i]['particle_type']}\n")
        
        if frame_num % 50 == 0:  # 50프레임마다만 출력
            print(f"프레임 {frame_num} 데이터가 {csv_filename}에 추가되었습니다. (입자 수: {len(positions):,}개)")

def load_initial_positions_from_csv(csv_filename):
    """CSV 파일에서 초기 입자 위치를 로드하는 함수"""
    try:
        if not os.path.exists(csv_filename):
            print(f"CSV 파일이 존재하지 않습니다: {csv_filename}")
            return None
        
        df = pd.read_csv(csv_filename)
        if len(df) == 0:
            print("CSV 파일이 비어있습니다.")
            return None
        
        # 위치 데이터 추출 (컬럼명에 따라 조정 필요)
        positions = []
        for _, row in df.iterrows():
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                positions.append([row['x'], row['y'], row['z']])
            else:
                # 기본 컬럼명 사용
                positions.append([row.iloc[0], row.iloc[1], row.iloc[2]])
        
        print(f"CSV에서 {len(positions)}개 입자 위치 로드 완료")
        return positions
        
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        return None

def load_particles_from_csv(csv_filename, target_frame):
    """특정 프레임의 입자 데이터를 CSV에서 로드 (메모리 효율적)"""
    try:
        positions, velocities, masses = [], [], []
        with open(csv_filename, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # 헤더만 있거나 빈 파일
                print(f"프레임 {target_frame} 데이터를 찾을 수 없습니다.")
                return None, None, None
            
            # 헤더 건너뛰기
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 8 and parts[0] == str(target_frame):
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
                    mass = float(parts[8])
                    positions.append([x, y, z])
                    velocities.append([vx, vy, vz])
                    masses.append(mass)
        
        if len(positions) == 0:
            print(f"프레임 {target_frame} 데이터를 찾을 수 없습니다.")
            return None, None, None
        
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        masses = np.array(masses, dtype=np.float32)
        
        print(f"프레임 {target_frame} 데이터 로드 완료: {len(positions)}개 입자")
        return positions, velocities, masses
        
    except FileNotFoundError:
        print(f"CSV 파일 {csv_filename}을 찾을 수 없습니다.")
        return None, None, None
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None

def find_last_saved_frame(csv_filename):
    """CSV 파일에서 마지막으로 저장된 프레임 번호 찾기 (메모리 효율적)"""
    try:
        # 메모리 효율적으로 마지막 프레임만 확인
        with open(csv_filename, 'r') as f:
            # 파일의 마지막 몇 줄만 읽어서 마지막 프레임 확인
            lines = f.readlines()
            if len(lines) <= 1:  # 헤더만 있거나 빈 파일
                return -1
            
            # 마지막 줄에서 프레임 번호 추출
            last_line = lines[-1].strip()
            if last_line:
                try:
                    last_frame = int(last_line.split(',')[0])
                    print(f"마지막 저장된 프레임: {last_frame}")
                    return last_frame
                except (ValueError, IndexError):
                    return -1
            return -1
    except FileNotFoundError:
        print(f"CSV 파일 {csv_filename}을 찾을 수 없습니다. 새로 시작합니다.")
        return -1
    except Exception as e:
        print(f"마지막 프레임 확인 중 오류 발생: {e}")
        return -1

def main():
    print("=== 시뮬레이션 초기화 시작 ===")
    print("진행률: 5% - Taichi 초기화 및 기본 설정 완료")
    
    # 메모리 상태 출력
    print("\n=== 메모리 상태 확인 ===")
    print_memory_status()
    print("=" * 30)
    
    # CSV 파일명 정의
    csv_filename = 'particle_simulation_datainterpolation.csv'
    # GAN 생성자 저장 경로 (학습 스크립트 파일이 위치한 폴더)
    gan_model_path = r"C:\\Users\\sunma\\OneDrive\\Desktop\\중요한 코딩들\\final_gan_generator.pth"
    
    print("진행률: 10% - CSV 파일명 설정 완료")
    
    # 기존 데이터 확인
    last_frame = find_last_saved_frame(csv_filename)
    
    print("진행률: 15% - 기존 데이터 확인 완료")
    
    if last_frame >= 0:
        print(f"\n=== 기존 데이터 발견 ===")
        print(f"마지막 저장된 프레임: {last_frame}")
        print(f"총 저장된 프레임 수: {last_frame + 1}")
        print("=" * 30)
        
        # 사용자 선택
        while True:
            choice = input("시뮬레이션을 어떻게 시작하시겠습니까?\n1. 기존 데이터에서 이어서 (프레임 {})\n2. 새로 시작\n선택 (1 또는 2): ".format(last_frame + 1))
            
            if choice == '1':
                print(f"\n기존 데이터에서 프레임 {last_frame + 1}부터 이어서 시작합니다...")
                print("진행률: 20% - 기존 데이터 로드 시작")
                # 기존 데이터 로드
                loaded_positions, loaded_velocities, loaded_masses = load_particles_from_csv(csv_filename, last_frame)
                
                if loaded_positions is not None:
                    print("진행률: 25% - 기존 데이터 로드 완료")
                    # 로드된 데이터로 시뮬레이션 초기화
                    N_all = len(loaded_positions)
                    print(f"로드된 입자 수: {N_all:,}개")
                    
                    # Taichi 필드에 데이터 설정
                    for i in range(N_all):
                        pos[i] = ti.Vector(loaded_positions[i])
                        vel[i] = ti.Vector(loaded_velocities[i])
                        mass[i] = loaded_masses[i]
                    
                    # 색상 배열 재생성 (기본 색상으로)
                    color_array = []
                    for i in range(N_all):
                        color_array.append(0x0080FF)  # 기본 파란색
                    
                    # 시뮬레이션 루프에서 사용할 변수들 정의
                    M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 105000, 0.3  # 기본 입자 10.5만개 (전체 15만개의 70%)
                    # 지구:테이아 = 10:1 비율로 입자 분배
                    N_earth = int(N_total / (1 + M_theia / M_earth))  # 105,000 / 1.1 = 95,455개
                    N_theia = N_total - N_earth  # 105,000 - 95,455 = 9,545개
                    N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
                    N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
                    R_earth, R_theia = 0.24, 0.14
                    center_earth = np.array([0.0, 0.0, 0.0])
                    center_theia = np.array([0.65, 0.0, 0.0])
                    # 4km/sec에 해당하는 초기 속도만 설정 (0.000151 시뮬레이션 단위)
                    # 45도 각도로 지구 중심 방향 (x축과 45도)
                    orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
                    v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
                    
                    # Earth 입자 수는 N_earth로 사용
                    
                    frame = last_frame + 1
                    print(f"시뮬레이션을 프레임 {frame}부터 재개합니다.")
                    print("진행률: 30% - 기존 데이터 기반 시뮬레이션 준비 완료")
                    break
                else:
                    print("데이터 로드에 실패했습니다. 새로 시작합니다.")
                    choice = '2'  # 새로 시작으로 변경
                    
            elif choice == '2':
                print("\n새로운 시뮬레이션을 시작합니다...")
                print("진행률: 20% - 새 시뮬레이션 초기화 시작")
                # 새 시뮬레이션 초기화
                frame = 0
                M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 105000, 0.3  # 기본 입자 10.5만개 (전체 15만개의 70%)
                # 지구:테이아 = 10:1 비율로 입자 분배
                N_earth = int(N_total / (1 + M_theia / M_earth))  # 105,000 / 1.1 = 95,455개
                N_theia = N_total - N_earth  # 105,000 - 95,455 = 9,545개
                N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
                N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
                R_earth, R_theia = 0.24, 0.14
                center_earth = np.array([0.0, 0.0, 0.0])
                center_theia = np.array([1.5, 0.0, 0.0])
                # 4km/sec에 해당하는 초기 속도만 설정 (0.000151 시뮬레이션 단위)
                # 45도 각도로 지구 중심 방향 (x축과 45도)
                orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
                v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
                
                print("진행률: 25% - 시뮬레이션 파라미터 설정 완료")
                # CSV 파일에서 초기 위치 로드
                initial_positions = load_initial_positions_from_csv("C:/Users/sunma/particles_simulation_data.csv")
                
                if initial_positions is not None:
                    print("진행률: 30% - CSV 파일에서 초기 위치 로드 완료")
                    print(f"CSV 파일에서 초기 위치를 로드했습니다: {len(initial_positions)}개 입자")
                    
                    print("진행률: 35% - 기본 입자 생성 시작")
                    # 기본 입자 생성 (위치만 CSV에서 가져오고, 속도와 질량은 기존 방식)
                    pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
                    pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
                    
                    # 디버깅: 테이아 입자 생성 확인
                    print(f"지구 입자 생성: {len(pe)}개 (코어: {N_earth_core}개, 맨틀: {N_earth_mantle}개)")
                    print(f"테이아 입자 생성: {len(pt)}개 (코어: {N_theia_core}개, 맨틀: {N_theia_mantle}개)")
                    print(f"테이아 중심 위치: {center_theia}, 반지름: {R_theia}")
                    
                    # CSV의 위치 정보로 기본 입자 위치 교체
                    basic_particle_count = len(pe) + len(pt)
                    if len(initial_positions) >= basic_particle_count:
                        # 기본 입자들의 위치를 CSV 위치로 교체
                        for i in range(len(pe)):
                            pe[i] = initial_positions[i]
                        for i in range(len(pt)):
                            pt[i] = initial_positions[len(pe) + i]
                    
                    all_pos = pe + pt
                    all_vel = ve + vt
                    all_mass = me + mt
                    color_array = []
                    for i in range(len(pe)):
                        pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                        color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
                    for i in range(len(pt)):
                        idx = len(pe) + i
                        pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                        color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)  # 테이아 색상 (코어: 갈색, 맨틀: 파란색)
                    N_all = len(all_pos)
                    
                    print("진행률: 40% - 기본 입자 설정 완료")
                    # 보간 입자 생성 (30% 비율로 고정)
                    interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
                    
                    # 보간 입자들의 위치를 CSV의 나머지 위치로 교체
                    if len(initial_positions) > basic_particle_count:
                        remaining_positions = initial_positions[basic_particle_count:]
                        for i, (interp_pos, interp_vel) in enumerate(interpolated):
                            if i < len(remaining_positions):
                                interp_pos = remaining_positions[i]
                            idx = N_all + i
                            pos[idx] = ti.Vector(interp_pos)
                            vel[idx] = ti.Vector(interp_vel)
                            mass[idx] = all_mass[0] * 0.8
                            color_array.append(0x00FF00)
                    else:
                        # CSV에 보간 입자 위치가 없으면 기존 방식 사용
                        for i, (interp_pos, interp_vel) in enumerate(interpolated):
                            idx = N_all + i
                            pos[idx] = ti.Vector(interp_pos)
                            vel[idx] = ti.Vector(interp_vel)
                            mass[idx] = all_mass[0] * 0.8
                            color_array.append(0x00FF00)
                    
                    # 총 입자 수 업데이트 (기본 + 보간)
                    N_all = len(all_pos) + len(interpolated)
                    
                    print("진행률: 45% - 보간 입자 생성 완료")
                    
                    # 생성된 입자 수 출력
                    print(f"=== 입자 생성 완료 ===")
                    print(f"기본 입자 수: {len(all_pos):,}개")
                    print(f"보간 입자 수: {len(interpolated):,}개")
                    print(f"총 입자 수: {N_all:,}개")
                    print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
                    print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
                    print("=====================")
                    
                    print("진행률: 50% - 초기 상태 CSV 저장 시작")
                    # 초기 상태를 CSV로 저장
                    initial_positions = pos.to_numpy()[:N_all]
                    initial_velocities = vel.to_numpy()[:N_all]
                    initial_masses = mass.to_numpy()[:N_all]
                    # save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
                else:
                    print("CSV 파일에서 초기 위치를 로드할 수 없어서 기존 방식으로 진행합니다.")
                    # 기존 방식으로 진행
                    pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
                    pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
                    
                    # 디버깅: 테이아 입자 생성 확인
                    print(f"지구 입자 생성: {len(pe)}개 (코어: {N_earth_core}개, 맨틀: {N_earth_mantle}개)")
                    print(f"테이아 입자 생성: {len(pt)}개 (코어: {N_theia_core}개, 맨틀: {N_theia_mantle}개)")
                    print(f"테이아 중심 위치: {center_theia}, 반지름: {R_theia}")
                    all_pos = pe + pt
                    all_vel = ve + vt
                    all_mass = me + mt
                    color_array = []
                    for i in range(len(pe)):
                        pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                        color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
                    for i in range(len(pt)):
                        idx = len(pe) + i
                        pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                        color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)  # 테이아 색상 (코어: 갈색, 맨틀: 파란색)
                    N_all = len(all_pos)
                    
                    # 보간 입자 생성 (30% 비율로 고정)
                    interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
                    
                    # 보간 입자들을 시뮬레이션에 추가
                    for i, (interp_pos, interp_vel) in enumerate(interpolated):
                        idx = N_all + i
                        pos[idx] = ti.Vector(interp_pos)
                        vel[idx] = ti.Vector(interp_vel)
                        mass[idx] = all_mass[0] * 0.8
                        color_array.append(0x00FF00)
                    
                    # 총 입자 수 업데이트 (기본 + 보간)
                    N_all = len(all_pos) + len(interpolated)
                    
                    # 생성된 입자 수 출력
                    print(f"=== 입자 생성 완료 ===")
                    print(f"기본 입자 수: {len(all_pos):,}개")
                    print(f"보간 입자 수: {len(interpolated):,}개")
                    print(f"총 입자 수: {N_all:,}개")
                    print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
                    print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
                    print("=====================")
                    
                    # 초기 상태를 CSV로 저장
                    initial_positions = pos.to_numpy()[:N_all]
                    initial_velocities = vel.to_numpy()[:N_all]
                    initial_masses = mass.to_numpy()[:N_all]
                    # save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
                break
            else:
                print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
    else:
        # 기존 데이터가 없는 경우 새로 시작
        print("\n기존 데이터가 없습니다. 새로운 시뮬레이션을 시작합니다...")
        frame = 0
        M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 105000, 0.3  # 기본 입자 10.5만개 (전체 15만개의 70%)
        # 지구:테이아 = 10:1 비율로 입자 분배
        N_earth = int(N_total / (1 + M_theia / M_earth))  # 105,000 / 1.1 = 95,455개
        N_theia = N_total - N_earth  # 105,000 - 95,455 = 9,545개
        N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
        N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
        R_earth, R_theia = 0.24, 0.14
        center_earth = np.array([0.0, 0.0, 0.0])
        center_theia = np.array([1.5, 0.0, 0.0])
        # 4km/sec에 해당하는 초기 속도만 설정 (0.000151 시뮬레이션 단위)
        # 45도 각도로 지구 중심 방향 (x축과 45도)
        orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
        v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
        
        # CSV 파일에서 초기 위치 로드
        initial_positions = load_initial_positions_from_csv("C:/Users/sunma/particles_simulation_data.csv")
        
        if initial_positions is not None:
            print(f"CSV 파일에서 초기 위치를 로드했습니다: {len(initial_positions)}개 입자")
            
            # 기본 입자 생성 (위치만 CSV에서 가져오고, 속도와 질량은 기존 방식)
            pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
            pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
            
            # CSV의 위치 정보로 기본 입자 위치 교체
            basic_particle_count = len(pe) + len(pt)
            if len(initial_positions) >= basic_particle_count:
                # 기본 입자들의 위치를 CSV 위치로 교체
                for i in range(len(pe)):
                    pe[i] = initial_positions[i]
                for i in range(len(pt)):
                    pt[i] = initial_positions[len(pe) + i]
            
            all_pos = pe + pt
            all_vel = ve + vt
            all_mass = me + mt
            color_array = []
            for i in range(len(pe)):
                pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
            for i in range(len(pt)):
                idx = len(pe) + i
                pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)
            N_all = len(all_pos)
            
            # 보간 입자 생성 (30% 비율로 고정)
            interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
            
            # 보간 입자들의 위치를 CSV의 나머지 위치로 교체
            if len(initial_positions) > basic_particle_count:
                remaining_positions = initial_positions[basic_particle_count:]
                for i, (interp_pos, interp_vel) in enumerate(interpolated):
                    if i < len(remaining_positions):
                        interp_pos = remaining_positions[i]
                    idx = N_all + i
                    pos[idx] = ti.Vector(interp_pos)
                    vel[idx] = ti.Vector(interp_vel)
                    mass[idx] = all_mass[0] * 0.8
                    color_array.append(0x00FF00)
            else:
                # CSV에 보간 입자 위치가 없으면 기존 방식 사용
                for i, (interp_pos, interp_vel) in enumerate(interpolated):
                    idx = N_all + i
                    pos[idx] = ti.Vector(interp_pos)
                    vel[idx] = ti.Vector(interp_vel)
                    mass[idx] = all_mass[0] * 0.8
                    color_array.append(0x00FF00)
            
            # 총 입자 수 업데이트 (기본 + 보간)
            N_all = len(all_pos) + len(interpolated)
            
            # 생성된 입자 수 출력
            print(f"=== 입자 생성 완료 ===")
            print(f"기본 입자 수: {len(all_pos):,}개")
            print(f"보간 입자 수: {len(interpolated):,}개")
            print(f"총 입자 수: {N_all:,}개")
            print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
            print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
            print("=====================")
            
            # 초기 상태를 CSV로 저장
            initial_positions = pos.to_numpy()[:N_all]
            initial_velocities = vel.to_numpy()[:N_all]
            initial_masses = mass.to_numpy()[:N_all]
            save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
        else:
            print("CSV 파일에서 초기 위치를 로드할 수 없어서 기존 방식으로 진행합니다.")
            # 기존 방식으로 진행
            pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
            pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
            all_pos = pe + pt
            all_vel = ve + vt
            all_mass = me + mt
            color_array = []
            for i in range(len(pe)):
                pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
            for i in range(len(pt)):
                idx = len(pe) + i
                pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)
            N_all = len(all_pos)
            
            # 보간 입자 생성 (30% 비율로 고정)
            interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
            
            # 보간 입자들을 시뮬레이션에 추가
            for i, (interp_pos, interp_vel) in enumerate(interpolated):
                idx = N_all + i
                pos[idx] = ti.Vector(interp_pos)
                vel[idx] = ti.Vector(interp_vel)
                mass[idx] = all_mass[0] * 0.8
                color_array.append(0x00FF00)
            
            # 총 입자 수 업데이트 (기본 + 보간)
            N_all = len(all_pos) + len(interpolated)
            
            # 생성된 입자 수 출력
            print(f"=== 입자 생성 완료 ===")
            print(f"기본 입자 수: {len(all_pos):,}개")
            print(f"보간 입자 수: {len(interpolated):,}개")
            print(f"총 입자 수: {N_all:,}개")
            print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
            print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
            print("=====================")
            
            # 초기 상태를 CSV로 저장
            initial_positions = pos.to_numpy()[:N_all]
            initial_velocities = vel.to_numpy()[:N_all]
            initial_masses = mass.to_numpy()[:N_all]
            save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
    
    gui = ti.GUI("Earth-Theia (With Interpolation)", res=(1600, 1000), background_color=0x000000, show_gui=True)
    
    # 스크린샷 저장 폴더 생성
    screenshot_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"스크린샷 저장 폴더: {screenshot_dir}")
    
    # all_pos가 정의되지 않은 경우를 위한 기본값 설정
    if 'all_pos' not in locals():
        print("경고: all_pos가 정의되지 않았습니다. 기본값을 설정합니다.")
        all_pos = []
    
    print("=== 시뮬레이션 루프 시작 ===")
    print(f"총 입자 수: {N_all:,}개")
    print(f"시작 프레임: {frame}")
    print("=" * 30)
    
    while gui.running and frame < 100000000:
        if frame % 100 == 0:  # 100프레임마다 상태 출력
            print(f"프레임 {frame}: 시뮬레이션 진행 중...")
            # 메모리 상태 확인 (100프레임마다)
            print_memory_status()
            # 메모리 정리 (필요시)
            if get_memory_usage() > 8.0:  # 8GB 이상 사용시 정리
                print("메모리 사용량이 높습니다. 정리를 시작합니다...")
                cleanup_memory()
        
        compute_gravity(N_all)
        apply_cohesion(N_all, 0.5, 0.4)  # 응집력과 범위를 증가
        clamp_escape_particles(N_earth, ti.Vector(center_earth.tolist(), dt=ti.f32), 0.5)  # 이탈 범위 확대
        reinforce_core(N_earth, ti.Vector(center_earth.tolist(), dt=ti.f32), 100.0, 0.05)
        update(N_all, DT)
        for _ in range(10):
            correct_overlap(N_all, PARTICLE_RADIUS_CORE, PARTICLE_RADIUS_MANTLE)
        if frame == 800:
            print(f"프레임 {frame}: 궤도 가이드 적용")
            guide_orbital_particles(N_all, ti.Vector(center_earth.tolist(), dt=ti.f32), 0.32, G * M_earth)
        
        # 매 프레임마다 입자 위치를 CSV로 저장
        current_positions = pos.to_numpy()[:N_all]
        current_velocities = vel.to_numpy()[:N_all]
        current_masses = mass.to_numpy()[:N_all]
        # save_particles_to_csv(current_positions, current_velocities, current_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
        
        np_pos = pos.to_numpy()[:N_all]
        # 3D 좌표를 2D로 투영 (카메라 회전 적용)
        camera_angle = frame * 0.01
        cos_a, sin_a = np.cos(camera_angle), np.sin(camera_angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        
        # 3D 좌표를 회전하고 2D로 투영 (지구와 테이아 모두 보이도록)
        # 카메라 중심을 지구와 테이아의 중간점으로 설정
        camera_center = [0.75, 0.0, 0.0]  # 지구(0,0,0)와 테이아(1.5,0,0)의 중간
        pos_3d = np_pos - camera_center
        pos_rotated = np.dot(pos_3d, rotation_matrix.T)
        pos_2d = pos_rotated[:, :2] / 1.0 + [0.5, 0.5]  # 스케일을 1.5로 줄여서 더 넓은 시야
        
        # 입자 크기를 1.5로 설정
        radii = np.full(N_all, 1.5)  # 모든 입자 크기를 1.5로 통일
        
        # 디버깅: 테이아 입자 위치 확인 (100프레임마다)
        if frame % 100 == 0:
            theia_particles = np_pos[len(pe):len(pe)+len(pt)] if 'pe' in locals() else np_pos[50000:55000]
            if len(theia_particles) > 0:
                theia_center = np.mean(theia_particles, axis=0)
                print(f"프레임 {frame}: 테이아 입자 중심 위치 = {theia_center}")
                print(f"테이아 입자 수: {len(theia_particles)}개")
        
        gui.circles(pos_2d, radius=radii, color=np.array(color_array))
        
        # 10프레임마다 스크린샷 저장
        if frame % 10 == 0:
            try:
                screenshot_path = os.path.join(screenshot_dir, f"frame_{frame:08d}.png")
                # Taichi GUI에서는 파일명을 인자로 전달하면 해당 경로로 저장됨
                gui.show(screenshot_path)
                print(f"프레임 {frame}: 스크린샷 저장 완료")
            except Exception as e:
                print(f"스크린샷 저장 실패: {e}")
                # 실패하면 일반 show()만 호출
                gui.show()
        else:
            gui.show()
        
        frame += 1
    
    print(f"=== 시뮬레이션 완료 ===")
    print(f"최종 프레임: {frame}")
    print("=" * 30)
    
    # 최종 상태를 CSV로 저장
    final_positions = pos.to_numpy()[:N_all]
    final_velocities = vel.to_numpy()[:N_all]
    final_masses = mass.to_numpy()[:N_all]
    # save_particles_to_csv(final_positions, final_velocities, final_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
    
    # 최종 메모리 상태 출력 및 정리
    print("\n=== 최종 메모리 상태 ===")
    print_memory_status()
    cleanup_memory()
    print("=" * 30)

if __name__ == '__main__':
    main()
