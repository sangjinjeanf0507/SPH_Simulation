import taichi as ti
import taichi.math as tm
import numpy as np
import torch
import pandas as pd
import os
from PIL import Image
import io
import psutil
import gc

ti.init(arch=ti.gpu, default_fp=ti.f32)

DIM = 3
MAX_PARTICLES = 200000
G = 3.568*10**-33
EPS = 0.05
DT = 0.05#0.0006

CFL = 0.25
SOFTENING = 0.1
DT_MAX = 0.001
DT_MIN = 1e-8
V_MAX = 1000.0
A_MAX = 10000.0

PARTICLE_RADIUS_MANTLE = 0.02
PARTICLE_RADIUS_CORE = PARTICLE_RADIUS_MANTLE * 4.0

pos = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
vel = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
acc = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
mass = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

density = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
pressure = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
h_smooth = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
internal_energy = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

v_max = ti.field(dtype=ti.f32, shape=())
a_max = ti.field(dtype=ti.f32, shape=())
dt_adaptive = ti.field(dtype=ti.f32, shape=())

GAMMA = 5.0 / 3.0
SIGMA_KERNEL = 1.0 / np.pi

# XSPH and viscosity parameters
XSPH_EPS = 0.25
NU = ti.field(dtype=ti.f32, shape=())
NU[None] = 0.01

@ti.func
def W_cubic_val(r: ti.f32, h: ti.f32) -> ti.f32:
    q = r / h
    alpha = 1.0 / (ti.math.pi * h * h * h)
    res = 0.0
    if q < 1.0:
        res = alpha * (1.0 - 1.5*q*q + 0.75*q*q*q)
    elif q < 2.0:
        t = 2.0 - q
        res = alpha * 0.25 * t * t * t
    return res

@ti.func
def dW_cubic_dr(r: ti.f32, h: ti.f32) -> ti.f32:
    q = r / h
    alpha = 1.0 / (ti.math.pi * h * h * h)
    result = 0.0
    if q < 1.0:
        result = alpha * (-3.0*q + 2.25*q*q) / h
    elif q < 2.0:
        t = 2.0 - q
        result = alpha * (-0.75 * t * t) / h
    return result

@ti.func
def W(r, h):
    q = r / h
    alpha = SIGMA_KERNEL / (h**DIM)
    result = 0.0
    if 0 <= q < 1:
        result = alpha * (1.0 - 1.5 * q**2 + 0.75 * q**3)
    elif 1 <= q < 2:
        result = alpha * (0.25 * (2.0 - q)**3)
    return result

@ti.func
def grad_W(r_vec, r, h):
    q = r / h
    alpha = SIGMA_KERNEL / (h**DIM)
    gradient_result = ti.Vector.zero(ti.f32, DIM)
    if r < 1e-9:
        pass
    else:
        dw_dq = 0.0
        if 0 <= q < 1:
            dw_dq = alpha * (-3.0 * q + 2.25 * q**2)
        elif 1 <= q < 2:
            dw_dq = alpha * (-0.75 * (2.0 - q)**2)
        gradient_result = dw_dq * r_vec / (r * h)
    return gradient_result

@ti.kernel
def compute_sph_density(n: ti.i32):
    for i in range(n):
        density[i] = mass[i] * W(0.0, h_smooth[i])
        for j in range(n):
            if i != j:
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
                if r < 2.0 * h_ij:
                    density[i] += mass[j] * W(r, h_ij)

@ti.kernel
def compute_sph_pressure(n: ti.i32):
    for i in range(n):
        if density[i] > 1e-9:
            pressure[i] = (GAMMA - 1.0) * density[i] * internal_energy[i]
        else:
            pressure[i] = 0.0

@ti.kernel
def compute_sph_forces(n: ti.i32):
    for i in range(n):
        a = ti.Vector.zero(ti.f32, DIM)
        
        for j in range(n):
            if i != j:
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
                
                if r < 1e-9 or r > 2.0 * h_ij:
                    continue
                    
                grad_Wij = grad_W(r_vec, r, h_ij)
                
                if density[i] > 1e-9 and density[j] > 1e-9:
                    a += -mass[j] * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2) * grad_Wij
                    v_dot_r = (vel[i] - vel[j]).dot(r_vec)
                    if v_dot_r < 0:
                        h_avg = (h_smooth[i] + h_smooth[j]) / 2.0
                        rho_avg = (density[i] + density[j]) / 2.0
                        ci = ti.sqrt(GAMMA * pressure[i] / density[i]) if density[i] > 1e-9 else 0.0
                        cj = ti.sqrt(GAMMA * pressure[j] / density[j]) if density[j] > 1e-9 else 0.0
                        c_avg = 0.5 * (ci + cj)
                        mu_ij = h_avg * v_dot_r / (r**2 + h_avg**2 * 0.01)
                        Pi_ij = (-0.1 * c_avg * mu_ij + 0.1 * mu_ij**2) / rho_avg
                        a += -mass[j] * Pi_ij * grad_Wij
        
        acc[i] += a

@ti.kernel
def sanitize_particles(n: ti.i32):
    for i in range(n):
        for d in range(DIM):
            if tm.isnan(pos[i][d]) or tm.isinf(pos[i][d]):
                pos[i][d] = 0.0
        
        for d in range(DIM):
            if tm.isnan(vel[i][d]) or tm.isinf(vel[i][d]):
                vel[i][d] = 0.0
        
        for d in range(DIM):
            if tm.isnan(acc[i][d]) or tm.isinf(acc[i][d]):
                acc[i][d] = 0.0
        
        if tm.isnan(density[i]) or tm.isinf(density[i]):
            density[i] = 1.0
        if tm.isnan(pressure[i]) or tm.isinf(pressure[i]):
            pressure[i] = 0.0
        if tm.isnan(h_smooth[i]) or tm.isinf(h_smooth[i]):
            h_smooth[i] = 0.1
        if tm.isnan(internal_energy[i]) or tm.isinf(internal_energy[i]):
            internal_energy[i] = 1.0

@ti.kernel
def clamp_velocities(n: ti.i32):
    for i in range(n):
        v_mag = vel[i].norm()
        if v_mag > V_MAX:
            vel[i] = vel[i].normalized() * V_MAX

@ti.kernel
def clamp_accelerations(n: ti.i32):
    for i in range(n):
        a_mag = acc[i].norm()
        if a_mag > A_MAX:
            acc[i] = acc[i].normalized() * A_MAX


@ti.kernel
def compute_adaptive_dt(n: ti.i32):
    v_max[None] = 0.0
    a_max[None] = 0.0
    
    for i in range(n):
        v_mag = vel[i].norm()
        a_mag = acc[i].norm()
        
        if v_mag > v_max[None]:
            v_max[None] = v_mag
        if a_mag > a_max[None]:
            a_max[None] = a_mag
    
    dt_v = CFL * SOFTENING / (v_max[None] + 1e-8)
    
    dt_a = CFL * ti.sqrt(SOFTENING / (a_max[None] + 1e-12))
    
    dt_adaptive[None] = ti.min(DT_MAX, ti.max(DT_MIN, ti.min(dt_v, dt_a)))

@ti.kernel
def apply_xsph(n: ti.i32, eps: ti.f32):
    for i in range(n):
        delta = ti.Vector.zero(ti.f32, DIM)
        for j in range(n):
            if i != j:
                rij = pos[j] - pos[i]
                r = ti.sqrt(rij.dot(rij))
                if r < 2.0 * h_smooth[i]:
                    w = W_cubic_val(r, h_smooth[i])
                    delta += mass[j] * (vel[j] - vel[i]) * w / (density[i] + density[j] + 1e-8)
        vel[i] += eps * delta

@ti.kernel
def accumulate_physical_viscosity(n: ti.i32):
    for i in range(n):
        a = ti.Vector.zero(ti.f32, DIM)
        for j in range(n):
            if i != j:
                h_avg = 0.5 * (h_smooth[i] + h_smooth[j])
                eps2 = (0.01 * h_avg) * (0.01 * h_avg)
                rij = pos[j] - pos[i]
                r2 = rij.dot(rij) + eps2
                r = ti.sqrt(r2)
                vij = vel[i] - vel[j]
                dwdr = dW_cubic_dr(r, h_avg)
                pij = vij.dot(rij) / r2
                gradW = (dwdr / (r + 1e-8)) * rij
                a += (2.0 * NU[None]) * mass[j] * pij * gradW / (density[i] * density[j] + 1e-8)
        acc[i] += a

@ti.kernel
def reset_acc(n: ti.i32):
    for i in range(n):
        acc[i] = ti.Vector.zero(ti.f32, DIM)

@ti.kernel
def compute_gravity(n: ti.i32):
    for i in range(n):
        for j in range(n):
            if i != j:
                r = pos[j] - pos[i]
                dist2 = r.dot(r) + EPS**2
                inv_r3 = 1.0 / (dist2 * ti.sqrt(dist2))
                acc[i] += G * mass[j] * r * inv_r3

@ti.kernel
def update(n: ti.i32, dt: ti.f32):
    for i in range(n):
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt

@ti.kernel
def update_leapfrog_second_half(n: ti.i32, dt: ti.f64):
    for i in range(n):
        vel[i] += 0.5 * acc[i] * dt

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
            vel[i] *= 0.8
            if r < max_dist * 1.5:
                toward_earth = (center - pos[i]).normalized()
                vel[i] += toward_earth * 0.1

@ti.kernel
def reinforce_core(n: ti.i32, center: ti.types.vector(3, ti.f32), strength: ti.f32, core_radius: ti.f32):
    for i in range(n):
        r = pos[i] - center
        dist = r.norm()
        if dist < core_radius:
            acc[i] += -r.normalized() * strength * (core_radius - dist)


def sample_solid_sphere(n, radius, center, dtype=np.float32):
    u = np.random.rand(n).astype(dtype)
    cost = np.random.uniform(-1.0, 1.0, size=n).astype(dtype)
    sint = np.sqrt(np.maximum(0.0, 1.0 - cost*cost)).astype(dtype)
    phi  = np.random.uniform(0.0, 2.0*np.pi, size=n).astype(dtype)

    r = (radius * np.cbrt(u)).astype(dtype)
    x = center[0] + r * sint * np.cos(phi)
    y = center[1] + r * sint * np.sin(phi)
    z = center[2] + r * cost
    pos = np.stack([x, y, z], axis=1).astype(dtype)
    return pos

def sample_hollow_shell(n, r_in, r_out, center, dtype=np.float32):
    u = np.random.rand(n).astype(dtype)
    r = np.cbrt((r_out**3 - r_in**3) * u + r_in**3).astype(dtype)
    cost = np.random.uniform(-1.0, 1.0, size=n).astype(dtype)
    sint = np.sqrt(np.maximum(0.0, 1.0 - cost*cost)).astype(dtype)
    phi  = np.random.uniform(0.0, 2.0*np.pi, size=n).astype(dtype)

    x = center[0] + r * sint * np.cos(phi)
    y = center[1] + r * sint * np.sin(phi)
    z = center[2] + r * cost
    return np.stack([x, y, z], axis=1).astype(dtype)

def sanitize_positions(name, pos, center_fallback=(0.0,0.0,0.0)):
    bad = ~np.isfinite(pos).all(axis=1)
    cnt = int(bad.sum())
    if cnt > 0:
        print(f"[SAN] {name}: non-finite {cnt}/{pos.shape[0]}")
        np.nan_to_num(pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        bad = ~np.isfinite(pos).all(axis=1)
        cnt = int(bad.sum())
        if cnt > 0:
            c = np.array(center_fallback, dtype=pos.dtype)
            repl = sample_solid_sphere(cnt, radius=1e-4, center=c, dtype=pos.dtype)
            pos[bad] = repl
    return pos

def report(name, arr):
    print(f"[CHK] {name}: shape={arr.shape}, finite={np.isfinite(arr).all()}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024**3)
    return memory_gb

def print_memory_status():
    memory_gb = get_memory_usage()
    print(f"현재 메모리 사용량: {memory_gb:.2f} GB")
    
    system_memory = psutil.virtual_memory()
    print(f"시스템 전체 메모리: {system_memory.total / (1024**3):.2f} GB")
    print(f"사용 가능한 메모리: {system_memory.available / (1024**3):.2f} GB")
    print(f"메모리 사용률: {system_memory.percent:.1f}%")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("메모리 정리 완료")

def initialize_body(center, N_core, N_mantle, R_body, core_fraction, M_total, velocity=np.zeros(3)):
    R_core = R_body * (core_fraction ** (1/3))
    M_core = (M_total * core_fraction) / N_core
    M_mantle = (M_total * (1 - core_fraction)) / N_mantle
    
    core_points = sample_solid_sphere(N_core, R_core * 0.7, center, dtype=np.float32)
    mantle_points = sample_hollow_shell(N_mantle, R_core * 0.7 + 0.005, R_body, center, dtype=np.float32)
    
    all_positions = np.vstack([core_points, mantle_points]).astype(np.float32)
    
    all_velocities = np.tile(velocity, (N_core + N_mantle, 1)).astype(np.float32)
    all_masses = np.concatenate([
        np.full(N_core, M_core, dtype=np.float32),
        np.full(N_mantle, M_mantle, dtype=np.float32)
    ])
    
    return all_positions, all_velocities, all_masses

@ti.kernel
def initialize_sph_fields(n: ti.i32):
    for i in range(n):
        h_smooth[i] = 0.1  # 초기 스무딩 길이
        internal_energy[i] = 1.0  # 초기 내부 에너지

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
    if len(positions) == 0:
        return []

    num_interpolated = int(len(positions) * interpolation_ratio)
    if num_interpolated <= 0:
        return []

    gen, device = _get_gan_generator(model_path)
    if gen is None:
        interpolated_particles = []
        for _ in range(num_interpolated):
            idx1 = np.random.randint(0, len(positions))
            idx2 = np.random.randint(0, len(positions))
            alpha = np.random.uniform(0.2, 0.8)
            interpolated_pos = positions[idx1] * alpha + positions[idx2] * (1 - alpha)
            interpolated_vel = velocities[idx1] * alpha + velocities[idx2] * (1 - alpha)
            interpolated_pos[2] = 0.0
            interpolated_vel[2] = 0.0
            noise_pos = np.random.normal(0, radius * 0.1, 3)
            noise_vel = np.random.normal(0, 0.01, 3)
            noise_pos[2] = 0.0
            noise_vel[2] = 0.0
            interpolated_particles.append((interpolated_pos + noise_pos, interpolated_vel + noise_vel))
        return interpolated_particles

    generated = []
    remaining = num_interpolated
    gen_particles = getattr(gen, "num_particles", 500)
    noise_dim = getattr(gen, "noise_dim", 100)
    seq_len_minus1 = 4

    positions_np = np.asarray(positions, dtype=np.float32)
    velocities_np = np.asarray(velocities, dtype=np.float32)

    while remaining > 0:
        if len(positions_np) >= gen_particles:
            sel_idx = np.random.choice(len(positions_np), size=gen_particles, replace=False)
        else:
            sel_idx = np.random.choice(len(positions_np), size=gen_particles, replace=True)
        cond_xy = positions_np[sel_idx, :2]
        cond_vxy = velocities_np[sel_idx, :2]
        last_frame = np.concatenate([cond_xy, cond_vxy], axis=1).astype(np.float32)  # (gen_particles, 4)

        input_sequence = np.stack([last_frame for _ in range(seq_len_minus1)], axis=0)  # (seq_len-1, P, 4)
        input_sequence = np.expand_dims(input_sequence, axis=0)  # (1, seq_len-1, P, 4)

        with torch.no_grad():
            inp = torch.from_numpy(input_sequence).to(device)
            noise = torch.randn(1, noise_dim, device=device)
            out = gen(inp, noise).cpu().numpy()[0]  # (P, 4)

        take = min(remaining, gen_particles)
        gen_xy = out[:take, 0:2]
        gen_vxy = out[:take, 2:4]

        gen_z = np.zeros((take, 1), dtype=np.float32)
        gen_vz = np.zeros((take, 1), dtype=np.float32)

        gen_pos = np.concatenate([gen_xy, gen_z], axis=1)
        gen_vel = np.concatenate([gen_vxy, gen_vz], axis=1)

        gen_pos += np.random.normal(0, radius * 0.05, gen_pos.shape).astype(np.float32)
        gen_vel += np.random.normal(0, 0.005, gen_vel.shape).astype(np.float32)

        for i in range(take):
            generated.append((gen_pos[i], gen_vel[i]))

        remaining -= take

    return generated

def save_particles_to_csv(positions, velocities, masses, frame_num, csv_filename, basic_particle_count=0, original_csv_filename=None):
    data = []
    
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
        particle_type = 'basic' if i < basic_particle_count else 'interpolated'
        
        current_pos = (positions[i, 0], positions[i, 1], positions[i, 2])
        
        if current_pos in position_to_id:
            particle_id = position_to_id[current_pos]
        else:
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
    try:
        if not os.path.exists(csv_filename):
            print(f"CSV 파일이 존재하지 않습니다: {csv_filename}")
            return None
        
        df = pd.read_csv(
            csv_filename,
            sep=",",
            engine="python",
            dtype="float32",
            na_values=["nan", "NaN", "inf", "-inf"],
            nrows=1000
        )
        
        if len(df) == 0:
            print("CSV 파일이 비어있습니다.")
            return None
        
        positions = []
        for _, row in df.iterrows():
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                x, y, z = row['x'], row['y'], row['z']
            else:
                x, y, z = row.iloc[0], row.iloc[1], row.iloc[2]
            
            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                positions.append([float(x), float(y), float(z)])
        
        print(f"CSV에서 {len(positions)}개 입자 위치 로드 완료")
        return positions
        
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        print("기본 방식으로 진행합니다.")
        return None

def load_particles_from_csv(csv_filename, target_frame):
    try:
        positions, velocities, masses = [], [], []
        
        try:
            df = pd.read_csv(
                csv_filename,
                sep=",",
                engine="python",
                dtype="str",
                na_values=["nan", "NaN", "inf", "-inf"],
                nrows=10000
            )
            
            target_data = df[df.iloc[:, 0] == str(target_frame)]
            
            for _, row in target_data.iterrows():
                if len(row) >= 8:
                    try:
                        x, y, z = float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
                        vx, vy, vz = float(row.iloc[5]), float(row.iloc[6]), float(row.iloc[7])
                        mass = float(row.iloc[8]) if len(row) > 8 else 1.0
                        
                        if pd.notna(x) and pd.notna(y) and pd.notna(z):
                            positions.append([x, y, z])
                            velocities.append([vx, vy, vz])
                            masses.append(mass)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as csv_error:
            print(f"pandas 로드 실패, 기본 방식으로 시도: {csv_error}")
            with open(csv_filename, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:
                    print(f"프레임 {target_frame} 데이터를 찾을 수 없습니다.")
                    return None, None, None
                
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 8 and parts[0] == str(target_frame):
                        try:
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
                            mass = float(parts[8])
                            positions.append([x, y, z])
                            velocities.append([vx, vy, vz])
                            masses.append(mass)
                        except (ValueError, IndexError):
                            continue
        
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
    try:
        try:
            df = pd.read_csv(
                csv_filename,
                sep=",",
                engine="python",
                dtype="str",
                na_values=["nan", "NaN", "inf", "-inf"],
                nrows=1000
            )
            
            if len(df) == 0:
                return -1
            
            last_frame = int(df.iloc[-1, 0])
            print(f"마지막 저장된 프레임: {last_frame}")
            return last_frame
            
        except Exception as pandas_error:
            print(f"pandas 로드 실패, 기본 방식으로 시도: {pandas_error}")
            with open(csv_filename, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:
                    return -1
                
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
    
    print("\n=== 메모리 상태 확인 ===")
    print_memory_status()
    print("=" * 30)
    
    csv_filename = 'particle_simulation_datainterpolation.csv'
    gan_model_path = r"C:\\Users\\sunma\\OneDrive\\Desktop\\중요한 코딩들\\final_gan_generator.pth"
    
    
    last_frame = find_last_saved_frame(csv_filename)
    
    
    if last_frame >= 0:
        print(f"\n=== 기존 데이터 발견 ===")
        print(f"마지막 저장된 프레임: {last_frame}")
        print(f"총 저장된 프레임 수: {last_frame + 1}")
        print("=" * 30)
        
        while True:
            choice = input("시뮬레이션을 어떻게 시작하시겠습니까?\n1. 기존 데이터에서 이어서 (프레임 {})\n2. 새로 시작\n선택 (1 또는 2): ".format(last_frame + 1))
            
            if choice == '1':
                print(f"\n기존 데이터에서 프레임 {last_frame + 1}부터 이어서 시작합니다...")
                # 기존 데이터 로드
                loaded_positions, loaded_velocities, loaded_masses = load_particles_from_csv(csv_filename, last_frame)
                
                if loaded_positions is not None:
                    N_all = len(loaded_positions)
                    print(f"로드된 입자 수: {N_all:,}개")
                    
                    for i in range(N_all):
                        pos[i] = ti.Vector(loaded_positions[i])
                        vel[i] = ti.Vector(loaded_velocities[i])
                        mass[i] = loaded_masses[i]
                    
                    initialize_sph_fields(N_all)
                    
                    color_array = []
                    for i in range(N_all):
                        color_array.append(0x0080FF)  # 기본 파란색
                    
                    M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 105000, 0.3  # 기본 입자 10.5만개 (참고 파일과 동일)
                    N_earth = int(N_total / (1 + M_theia / M_earth))  # 105000 / 1.1 = 95455개
                    N_theia = N_total - N_earth  # 105000 - 95455 = 9545개
                    N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
                    N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
                    R_earth, R_theia = 0.24, 0.14
                    center_earth = np.array([0.0, 0.0, 0.0])
                    center_theia = np.array([0.65, 0.0, 0.0])
                    orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
                    v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
                    
                    
                    frame = last_frame + 1
                    print(f"시뮬레이션을 프레임 {frame}부터 재개합니다.")
                    break
                else:
                    print("데이터 로드에 실패했습니다. 새로 시작합니다.")
                    choice = '2'  # 새로 시작으로 변경
                    
            elif choice == '2':
                print("\n새로운 시뮬레이션을 시작합니다...")
                frame = 0
                M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 105000, 0.3  # 기본 입자 10.5만개 (참고 파일과 동일)
                N_earth = int(N_total / (1 + M_theia / M_earth))  # 105000 / 1.1 = 95455개
                N_theia = N_total - N_earth  # 105000 - 95455 = 9545개
                N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
                N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
                R_earth, R_theia = 0.24, 0.14
                center_earth = np.array([0.0, 0.0, 0.0])
                center_theia = np.array([1.5, 0.0, 0.0])
                orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
                v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
                
                initial_positions = load_initial_positions_from_csv("C:/Users/sunma/particles_simulation_data.csv")
                
                if initial_positions is not None:
                    print(f"CSV 파일에서 초기 위치를 로드했습니다: {len(initial_positions)}개 입자")
                    
                    pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
                    pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
                    
                    print(f"지구 입자 생성: {len(pe)}개 (코어: {N_earth_core}개, 맨틀: {N_earth_mantle}개)")
                    print(f"테이아 입자 생성: {len(pt)}개 (코어: {N_theia_core}개, 맨틀: {N_theia_mantle}개)")
                    print(f"테이아 중심 위치: {center_theia}, 반지름: {R_theia}")
                    
                    basic_particle_count = len(pe) + len(pt)
                    if len(initial_positions) >= basic_particle_count:
                        for i in range(len(pe)):
                            pe[i] = initial_positions[i]
                        for i in range(len(pt)):
                            pt[i] = initial_positions[len(pe) + i]
                    
                    pe = sanitize_positions("Earth", pe, center_fallback=(0,0,0))
                    pt = sanitize_positions("Theia", pt, center_fallback=(1.5,0,0))
                    
                    report("pe", pe)
                    report("pt", pt)
                    
                    all_pos = np.concatenate([pe, pt], axis=0)
                    all_vel = np.concatenate([ve, vt], axis=0)
                    all_mass = np.concatenate([me, mt], axis=0)
                    color_array = []
                    for i in range(len(pe)):
                        pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                        color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
                    for i in range(len(pt)):
                        idx = len(pe) + i
                        pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                        color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)  # 테이아 색상 (코어: 갈색, 맨틀: 파란색)
                    N_all = len(all_pos)
                    
                    initialize_sph_fields(N_all)
                    
                    interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
                    
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
                        for i, (interp_pos, interp_vel) in enumerate(interpolated):
                            idx = N_all + i
                            pos[idx] = ti.Vector(interp_pos)
                            vel[idx] = ti.Vector(interp_vel)
                            mass[idx] = all_mass[0] * 0.8
                            color_array.append(0x00FF00)
                    
                    N_all = len(all_pos) + len(interpolated)
                    
                    initialize_sph_fields(N_all)
                    
                    
                    # 생성된 입자 수 출력
                    print(f"=== 입자 생성 완료 ===")
                    print(f"기본 입자 수: {len(all_pos):,}개")
                    print(f"보간 입자 수: {len(interpolated):,}개")
                    print(f"총 입자 수: {N_all:,}개")
                    print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
                    print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
                    print("=====================")
                    
                    initial_positions = pos.to_numpy()[:N_all]
                    initial_velocities = vel.to_numpy()[:N_all]
                    initial_masses = mass.to_numpy()[:N_all]
                    # save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
                else:
                    print("CSV 파일에서 초기 위치를 로드할 수 없어서 기존 방식으로 진행합니다.")
                    pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
                    pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
                    
                    print(f"지구 입자 생성: {len(pe)}개 (코어: {N_earth_core}개, 맨틀: {N_earth_mantle}개)")
                    print(f"테이아 입자 생성: {len(pt)}개 (코어: {N_theia_core}개, 맨틀: {N_theia_mantle}개)")
                    print(f"테이아 중심 위치: {center_theia}, 반지름: {R_theia}")
                    all_pos = np.concatenate([pe, pt], axis=0)
                    all_vel = np.concatenate([ve, vt], axis=0)
                    all_mass = np.concatenate([me, mt], axis=0)
                    color_array = []
                    for i in range(len(pe)):
                        pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                        color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
                    for i in range(len(pt)):
                        idx = len(pe) + i
                        pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                        color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)  # 테이아 색상 (코어: 갈색, 맨틀: 파란색)
                    N_all = len(all_pos)
                    
                    interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
                    
                    for i, (interp_pos, interp_vel) in enumerate(interpolated):
                        idx = N_all + i
                        pos[idx] = ti.Vector(interp_pos)
                        vel[idx] = ti.Vector(interp_vel)
                        mass[idx] = all_mass[0] * 0.8
                        color_array.append(0x00FF00)
                    
                    N_all = len(all_pos) + len(interpolated)
                    
                    initialize_sph_fields(N_all)
                    
                    # 생성된 입자 수 출력
                    print(f"=== 입자 생성 완료 ===")
                    print(f"기본 입자 수: {len(all_pos):,}개")
                    print(f"보간 입자 수: {len(interpolated):,}개")
                    print(f"총 입자 수: {N_all:,}개")
                    print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
                    print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
                    print("=====================")
                    
                    initial_positions = pos.to_numpy()[:N_all]
                    initial_velocities = vel.to_numpy()[:N_all]
                    initial_masses = mass.to_numpy()[:N_all]
                    # save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
                break
            else:
                print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
    else:
        print("\n기존 데이터가 없습니다. 새로운 시뮬레이션을 시작합니다...")
        frame = 0
        M_earth, M_theia, N_total, core_fraction = 10.0, 1.0, 1000, 0.3  # 기본 입자 1000개
        N_earth = int(N_total / (1 + M_theia / M_earth))  # 105,000 / 1.1 = 95,455개
        N_theia = N_total - N_earth  # 105,000 - 95,455 = 9,545개
        N_earth_core = int(N_earth * 0.1); N_earth_mantle = N_earth - N_earth_core
        N_theia_core = int(N_theia * 0.1); N_theia_mantle = N_theia - N_theia_core
        R_earth, R_theia = 0.24, 0.14
        center_earth = np.array([0.0, 0.0, 0.0])
        center_theia = np.array([1.5, 0.0, 0.0])
        orbital_dir = np.array([-0.707, -0.707, 0.0])  # 45도 각도 (cos(45°)=0.707, sin(45°)=0.707)
        v_theia = 0.000151 * orbital_dir  # 초기 속도만 설정, 이후 물리 법칙에 맡김
        
        initial_positions = load_initial_positions_from_csv("C:/Users/sunma/particles_simulation_data.csv")
        
        if initial_positions is not None:
            print(f"CSV 파일에서 초기 위치를 로드했습니다: {len(initial_positions)}개 입자")
            
            pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
            pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
            
            basic_particle_count = len(pe) + len(pt)
            if len(initial_positions) >= basic_particle_count:
                for i in range(len(pe)):
                    pe[i] = initial_positions[i]
                for i in range(len(pt)):
                    pt[i] = initial_positions[len(pe) + i]
            
            all_pos = np.concatenate([pe, pt], axis=0)
            all_vel = np.concatenate([ve, vt], axis=0)
            all_mass = np.concatenate([me, mt], axis=0)
            color_array = []
            for i in range(len(pe)):
                pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
            for i in range(len(pt)):
                idx = len(pe) + i
                pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)
            N_all = len(all_pos)
            
            interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
            
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
                for i, (interp_pos, interp_vel) in enumerate(interpolated):
                    idx = N_all + i
                    pos[idx] = ti.Vector(interp_pos)
                    vel[idx] = ti.Vector(interp_vel)
                    mass[idx] = all_mass[0] * 0.8
                    color_array.append(0x00FF00)
            
            N_all = len(all_pos) + len(interpolated)
            
            initialize_sph_fields(N_all)
            
            # 생성된 입자 수 출력
            print(f"=== 입자 생성 완료 ===")
            print(f"기본 입자 수: {len(all_pos):,}개")
            print(f"보간 입자 수: {len(interpolated):,}개")
            print(f"총 입자 수: {N_all:,}개")
            print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
            print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
            print("=====================")
            
            initial_positions = pos.to_numpy()[:N_all]
            initial_velocities = vel.to_numpy()[:N_all]
            initial_masses = mass.to_numpy()[:N_all]
            save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
        else:
            print("CSV 파일에서 초기 위치를 로드할 수 없어서 기존 방식으로 진행합니다.")
            pe, ve, me = initialize_body(center_earth, N_earth_core, N_earth_mantle, R_earth, core_fraction, M_earth)
            pt, vt, mt = initialize_body(center_theia, N_theia_core, N_theia_mantle, R_theia, core_fraction, M_theia, v_theia)
            all_pos = np.concatenate([pe, pt], axis=0)
            all_vel = np.concatenate([ve, vt], axis=0)
            all_mass = np.concatenate([me, mt], axis=0)
            color_array = []
            for i in range(len(pe)):
                pos[i], vel[i], mass[i] = ti.Vector(pe[i]), ti.Vector(ve[i]), me[i]
                color_array.append(0xFF0000 if i < N_earth_core else 0x0080FF)
            for i in range(len(pt)):
                idx = len(pe) + i
                pos[idx], vel[idx], mass[idx] = ti.Vector(pt[i]), ti.Vector(vt[i]), mt[i]
                color_array.append(0x8B4513 if i < N_theia_core else 0x0000FF)
            N_all = len(all_pos)
            
            interpolated = interpolate_with_gan(gan_model_path, np.array(all_pos), np.array(all_vel), radius=0.01, interpolation_ratio=0.3)
            
            for i, (interp_pos, interp_vel) in enumerate(interpolated):
                idx = N_all + i
                pos[idx] = ti.Vector(interp_pos)
                vel[idx] = ti.Vector(interp_vel)
                mass[idx] = all_mass[0] * 0.8
                color_array.append(0x00FF00)
            
            N_all = len(all_pos) + len(interpolated)
            
            initialize_sph_fields(N_all)
            
            # 생성된 입자 수 출력
            print(f"=== 입자 생성 완료 ===")
            print(f"기본 입자 수: {len(all_pos):,}개")
            print(f"보간 입자 수: {len(interpolated):,}개")
            print(f"총 입자 수: {N_all:,}개")
            print(f"기본 입자 비율: {len(all_pos)/N_all*100:.1f}%")
            print(f"보간 입자 비율: {len(interpolated)/N_all*100:.1f}%")
            print("=====================")
            
            initial_positions = pos.to_numpy()[:N_all]
            initial_velocities = vel.to_numpy()[:N_all]
            initial_masses = mass.to_numpy()[:N_all]
            save_particles_to_csv(initial_positions, initial_velocities, initial_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
    
    gui = ti.GUI("Earth-Theia (With Interpolation)", res=(1600, 1000), background_color=0x000000, show_gui=True)
    
    screenshot_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"스크린샷 저장 폴더: {screenshot_dir}")
    
    if 'all_pos' not in locals():
        print("경고: all_pos가 정의되지 않았습니다. 기본값을 설정합니다.")
        all_pos = []
    
    print("=== 시뮬레이션 루프 시작 ===")
    print(f"총 입자 수: {N_all:,}개")
    print(f"시작 프레임: {frame}")
    print("=" * 30)
    
    while gui.running and frame < 100000000:
        sanitize_particles(N_all)
        compute_adaptive_dt(N_all)
        current_dt = min(DT, dt_adaptive[None])
        
        if frame % 100 == 0:
            print(f"프레임 {frame}: 시뮬레이션 진행 중...")
            print(f"  최대 속도: {v_max[None]:.6f}, 최대 가속도: {a_max[None]:.6f}")
            print(f"  적응형 dt: {dt_adaptive[None]:.8f}, 사용 dt: {current_dt:.8f}")
            
            pos_np = pos.to_numpy()[:N_all]
            if np.isnan(pos_np).any():
                print(f"  경고: 프레임 {frame}에서 NaN 발견!")
                break
            
            print_memory_status()
            if get_memory_usage() > 8.0:  # 8GB 이상 사용시 정리
                print("메모리 사용량이 높습니다. 정리를 시작합니다...")
                cleanup_memory()
        
        reset_acc(N_all)
        compute_sph_density(N_all)
        compute_sph_pressure(N_all)
        compute_sph_forces(N_all)
        accumulate_physical_viscosity(N_all)
        compute_gravity(N_all)
        apply_cohesion(N_all, 0.5, 0.4)
        clamp_escape_particles(N_earth, ti.Vector(center_earth.tolist()), 0.5)
        reinforce_core(N_earth, ti.Vector(center_earth.tolist()), 100.0, 0.05)
        
        clamp_accelerations(N_all)
        
        update(N_all, current_dt)
        
        clamp_velocities(N_all)
        apply_xsph(N_all, XSPH_EPS)
        for _ in range(10):
            correct_overlap(N_all, PARTICLE_RADIUS_CORE, PARTICLE_RADIUS_MANTLE)
        
        current_positions = pos.to_numpy()[:N_all]
        current_velocities = vel.to_numpy()[:N_all]
        current_masses = mass.to_numpy()[:N_all]
        # save_particles_to_csv(current_positions, current_velocities, current_masses, frame, csv_filename, len(all_pos), "C:/Users/sunma/particles_simulation_data.csv")
        
        np_pos = pos.to_numpy()[:N_all]
        camera_angle = frame * 0.01
        cos_a, sin_a = np.cos(camera_angle), np.sin(camera_angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        
        camera_center = [0.75, 0.0, 0.0]
        pos_3d = np_pos - camera_center
        pos_rotated = np.dot(pos_3d, rotation_matrix.T)
        pos_2d = pos_rotated[:, :2] / 15.0 + [0.5, 0.5]
        
        n = pos_2d.shape[0]
        radii = np.full(n, 1.5, dtype=np.float32)
        
        color_array = np.zeros(n, dtype=np.uint32)
        
        for i in range(N_earth_core):
            if i < n:
                color_array[i] = 0xFF0000
        
        for i in range(N_earth_core, N_earth):
            if i < n:
                color_array[i] = 0x0080FF
                
        for i in range(N_earth, N_earth + N_theia_core):
            if i < n:
                color_array[i] = 0x8B4513
                
        for i in range(N_earth + N_theia_core, N_earth + N_theia):
            if i < n:
                color_array[i] = 0x0000FF
        
        for i in range(N_earth + N_theia, n):
            color_array[i] = 0x00FF00
        
        gui.circles(pos_2d, radius=radii, color=color_array)
        
        if frame % 100 == 0:
            theia_particles = np_pos[len(pe):len(pe)+len(pt)] if 'pe' in locals() else np_pos[50000:55000]
            if len(theia_particles) > 0:
                theia_center = np.mean(theia_particles, axis=0)
                print(f"프레임 {frame}: 테이아 입자 중심 위치 = {theia_center}")
                print(f"테이아 입자 수: {len(theia_particles)}개")
        
        
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