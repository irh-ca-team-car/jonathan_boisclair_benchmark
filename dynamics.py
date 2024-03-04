

import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Callable, List
import time
import sys
import math

V_mass = 1500
Max_Mech_Brake = 6  # m/s2
Max_Mech_Acc = 2.65  # m/s2
Fric_coeff = 0.02  # 0.7
Gravity = 9.81
K_eff_battery = 0.5
Sub_Time_step = 0.1
Wheel_radius = 0.57


rho = 1.2
Cd = 0.3
A = 2.2

# Calculated constants
Max_Force_Mech_Brake = Max_Mech_Brake * V_mass


@dataclass
class State:
    v: float
    s: float
    a: float = 0.0
    f_m: float = 0.0
    f_t: float = 0.0
    f_r: float = 0.0
    f_l: float = 0.0


class Path:
    states: List[State]
    actions: List[float]

    def __init__(self, time_step) -> None:
        self.regenerated = 0
        self.consumed = 0
        self.aero = 0
        self.gravity = 0
        self.friction = 0
        self.braking_energy = 0
        self.states = []
        self.extra_states = []
        self.actions = []
        self.valid = False
        self.time_step = time_step

    def total_energy(self):
        return self.braking_energy + self.consumed + self.aero + self.gravity + self.friction

    @property
    def v(self):
        return [s.v for s in self.states]

    @property
    def a(self):
        return [s.a for s in self.states]

    @property
    def s(self):
        return [s.s for s in self.states]

    @property
    def t(self):
        t = 0
        ret = []
        for v in self.states:
            ret.append(t)
            t += self.time_step
        return ret

    @property
    def kinetic(self):
        return 0.5 * V_mass * pow(self.v[-1], 2)

    def through(self, action, max_regen=0.0, slope_fn: [Callable[[float], float]] = None, Time_step=5):
        state = self.states[-1]

        newly_regen = 0
        newly_lost = 0
        newly_consumed = 0
        newly_aero = 0
        newly_gravity = 0
        newly_friction = 0

        distance_traveled = 0
        new_speed = state.v
        t = 0
        extra_states = []
        Early_Stop = False
        while t < Time_step and (not Early_Stop):
            t += Sub_Time_step
            distance_traveled_at_constant_speed = Sub_Time_step * state.v
            rps = distance_traveled_at_constant_speed / Wheel_radius
            # Distance / CIrc = rps(nbr)
            #Distance / (2*pi*r)
            # rps(radian) = Distance / (2*pi*r) * 2*pi

            F_friction = Fric_coeff * V_mass * Gravity * \
                math.cos(slope_fn(state.s + distance_traveled))
            K_aero = rho * Cd * A

            F_gravity = V_mass * Gravity * \
                math.sin(slope_fn(state.s + distance_traveled))

            F_regen = max_regen / K_eff_battery / Wheel_radius / rps
            if action > 0 and action < F_regen:
                F_regen = action
            F_motor = 0
            F_mech = min(action - F_regen, Max_Force_Mech_Brake)

            if action < 0:
                F_regen = 0
                F_mech = 0
                F_motor = -action

            F_aero = 0.5 * K_aero * new_speed*new_speed
            Fv = F_motor - F_friction - F_aero - F_mech - F_regen - F_gravity
            a = Fv / V_mass
            distance_traveled_this_step = a * Sub_Time_step * \
                Sub_Time_step + new_speed*Sub_Time_step
            if distance_traveled_this_step < 0:
                distance_traveled_this_step = 0
                Early_Stop = True
            distance_traveled += abs(distance_traveled_this_step)
            prev_speed = new_speed
            new_speed = new_speed + a * Sub_Time_step

            distance_traveled_at_half_speed = Sub_Time_step * \
                (prev_speed + new_speed) / 2
            rps = distance_traveled_at_half_speed / Wheel_radius

            newly_regen += F_regen * rps * Wheel_radius
            newly_lost += F_mech * rps * Wheel_radius
            newly_consumed += F_motor * rps * Wheel_radius
            newly_gravity += F_gravity * rps * Wheel_radius
            newly_aero += F_aero * rps * Wheel_radius
            newly_friction += F_friction * rps * Wheel_radius

            new_extra_state = State(
                v=new_speed, s=state.s + distance_traveled, a=a, f_t=Fv, f_m=F_motor, f_r=F_regen, f_l=F_friction + F_aero)
            extra_states.append(new_extra_state)

        new_state = State(v=new_speed, s=state.s + distance_traveled, a=a)
        p = Path(self.time_step)
        p.regenerated = self.regenerated + newly_regen
        p.consumed = self.consumed + newly_consumed
        p.braking_energy = self.braking_energy + newly_lost + newly_regen
        p.aero = self.aero + newly_aero
        p.gravity = self.gravity + newly_gravity
        p.friction = self.friction + newly_friction
        p.states = [*self.states, new_state]
        p.extra_states = [*self.extra_states, *extra_states]
        p.actions = [*self.actions, action]

        return p

    def build(self):
        path = [self.states[0]]
        for i in range(len(self.actions)):
            path.append(self.actions[i])
            path.append(self.states[i+1])
        return path

    def __str__(self) -> str:
        return str(self.build())

    def validate_jerk(self, Time_step=5) -> bool:
        return True
        # 3 m/s3
        for i in range(1, len(self.states)):
            da = self.states[i].a - self.states[i-1].a
            jerk = da / Time_step
            if(abs(jerk) > 6):
                #print("REJECT","Jerk invalid")
                return False
        return True


def possible_braking_values(p: Path) -> List[float]:
    STEP = 0.25
    lst = []
    t = -Max_Mech_Acc
    while t <= Max_Mech_Brake:
        lst.append(t)
        t += STEP
    lst.append(Max_Mech_Brake)
    return lst


def flat_land(s: float) -> float:
    return 0


def vehicle_kinetic(speed):
    fake_path = Path(0)
    fake_path.states.append(State(v=speed, s=0))
    return fake_path.kinetic


def generate_optimal_profile(initial_speed,
                             target_distance,
                             target_speed=0,
                             max_regen=0.0,
                             Distance_tolerance=5,
                             Time_step=5,
                             generator: Callable[[Path], List[float]] = None,
                             slope_fn: Callable[[float], float] = None) -> List[Path]:
    paths: List[Path] = []
    if generator is None:
        generator = possible_braking_values
    if slope_fn is None:
        slope_fn = flat_land
    fake_path = Path(Time_step)
    fake_path.states.append(State(v=initial_speed, s=0))
    paths.append(fake_path)

    mode_brake = target_speed < initial_speed

    best_valid_path = None
    valid_paths = []
    i = 0
    while len([p for p in paths if not p.valid]) > 0:
        current_path: Path = paths[0]
        del paths[0]

        i += 1
        if i >= 5000:
            time.sleep(0.001)
            print("Slowing down to not hog up the cpu")
            i = 0
        if mode_brake and current_path.states[-1].v > initial_speed:
            continue
        if (not mode_brake) and current_path.states[-1].v < initial_speed:
            continue
        if current_path.states[-1].s > target_distance:
            # Ignore all path where it didn't stop in time
            #print("REJECT","Ignoring path that went further than distance")
            continue
        if current_path.valid:
            # Just move already completed path to the next
            #print("REJECT","Path already valid")
            paths.append(current_path)
            continue
        if mode_brake and current_path.states[-1].v <= 0:
            current_path.states[-1].v = 0

        if mode_brake and current_path.states[-1].v <= target_speed:
            current_path.states[-1].v = target_speed
            #print("ACCEPT","Brake successful")
            current_path.valid = True
            if current_path.states[-1].s + Distance_tolerance > target_distance:
                paths.append(current_path)
            continue
        if (not mode_brake) and current_path.states[-1].v >= target_speed:
            current_path.states[-1].v = target_speed
            #print("ACCEPT","Acceleration successful")
            current_path.valid = True
            if current_path.states[-1].s + Distance_tolerance > target_distance:
                paths.append(current_path)
            continue

        to_visit = generator(current_path)
        for deceleration in to_visit:
            n_path = current_path.through(
                deceleration * V_mass, max_regen=max_regen, slope_fn=slope_fn, Time_step=Time_step)
            if n_path.validate_jerk(Time_step):
                if n_path.states[-1].v <= 0:
                    n_path.states[-1].v = 0
                if (mode_brake and n_path.states[-1].v <= target_speed) or ((not mode_brake) and n_path.states[-1].v >= target_speed):
                    n_path.states[-1].v = target_speed
                    if n_path.states[-1].s + Distance_tolerance > target_distance:
                        n_path.valid = True
                    if best_valid_path is not None:
                        if mode_brake and n_path.regenerated + n_path.kinetic < best_valid_path.regenerated:
                            continue  # impossible to regenerate more
                # vehicle is stale, ignore
                if abs(n_path.states[-1].s - n_path.states[-2].s) < 0.1:
                    #print("REJECT","Vehicle is stale")
                    continue
                if n_path.states[-1].s < target_distance:
                    paths.insert(0, n_path)

        valid_paths = [p for p in paths if p.valid]
        if len(valid_paths) > 0:
            sorted(valid_paths, key=lambda a: (-a.regenerated +
                   a.consumed, a.braking_energy))
            best_valid_path = valid_paths[0]

    paths = [p for p in paths if p.valid]
    paths.sort(key=lambda a: (-a.regenerated + a.consumed, a.braking_energy))

    return paths


if False:
    t_s = 15/30.
    fake_path = Path(t_s)
    fake_path.states.append(State(v=10/3.6, s=0))
    kinetic_start = fake_path.kinetic
    new_path = fake_path.through(
        4.0*V_mass, 500000, slope_fn=lambda _: 0, Time_step=t_s)
    kinetic_end = new_path.kinetic

    print(new_path)
    p = (new_path.extra_states[0].f_t -
         new_path.extra_states[0].f_l) / new_path.extra_states[0].f_t
    x = p * 1/30. * (kinetic_end - kinetic_start) / 30.
    y = -1.1535*x - 1.33

    print("Energy, from regression,", y,
          "from_simulation", new_path.regenerated/3600)

# x = []
# spds = []
# b = []
#
# plt.plot(x, spds)
# plt.plot(x, b)
# plt.show()

# spds=[]
# for initial_speed in range(100):
#     initial_speed /= 3.6  # to m/s
#     best_braking_profiles = generate_optimal_profile(initial_speed=initial_speed, target_distance=200,target_speed=0,max_regen=3000,Distance_tolerance=10000000, Time_step=20)
#     sys.stdout.flush()
#     if(len(best_braking_profiles)>0):
#         print("Speed: "+str(int(initial_speed*3.6))+", Energy:"+str(best_braking_profiles[0].regenerated)+"J")
#         spds.append(best_braking_profiles[0].regenerated)
# import json
# print(json.dumps(spds))
# import matplotlib.pyplot as plt
# plt.plot(spds)
# plt.show()


# t1 = time.time()
# profiles = generate_optimal_profile(initial_speed=25,target_distance=40,target_speed=0,max_regen=5000, Distance_tolerance=10000000,Time_step=1)
# t2 = time.time()

# print("Generated in ",t2-t1,"s")
# print("Vehicle initial kinetic energy:",vehicle_kinetic(25))
# if len(profiles)>0:
#     print(len(profiles),"matching profiles, selecting best profile")
#     best_braking_profile:Path = profiles[0]
#     print("Consumed",best_braking_profile.consumed,"J", best_braking_profile.consumed/3600,"Wh")
#     print("Regenerated",best_braking_profile.regenerated,"J", best_braking_profile.regenerated/3600,"Wh")
#     print("Braking energy",best_braking_profile.braking_energy,"J", best_braking_profile.braking_energy/3600,"Wh")
#     print("Braking energy + loss",best_braking_profile.total_energy(),"J", best_braking_profile.total_energy()/3600,"Wh")

#     printable_profile= best_braking_profile.build()
#     printable_profile = [str(p/V_mass)+" m/s2" if isinstance(p,float) else str(p.v*3.6)+"km/h:"+str(p.s) for p in printable_profile]
#     print(printable_profile)

#     import matplotlib.pyplot as plt

#     plt.plot(best_braking_profile.s, best_braking_profile.v)
#     plt.show()
# else:
#     print("No profiles")

# print("{")
# for initial_speed in range(30):
#     initial_speed+=10
#     initial_speed /= 3.6  # to m/s
#     for distance_to_brake in range(1,100):
#         best_braking_profiles = dyn_prog(initial_speed, distance_to_brake,0)
#         sys.stdout.flush()
#         if(len(best_braking_profiles)>0):
#             print("\""+str(int(initial_speed*3.6)-10)+"\":"+str(distance_to_brake)+",")
#             break
# print("}")
# for distance_to_brake in range(10,102,1):
#     initial_speed = 100  # km/h
#     initial_speed /= 3.6  # to m/s
#     print("Generating profile for 100kmh to 0kmh in ", distance_to_brake,"m")
#     best_braking_profiles = generate_optimal_profile(initial_speed, distance_to_brake,5000)
#     sys.stdout.flush()
#     if len(best_braking_profiles) > 0:
#         print(len(best_braking_profiles),"matching profiles, selecting best profile")
#         best_braking_profile = best_braking_profiles[0]
#         print(distance_to_brake,"m Regenerated",best_braking_profile.regenerated,"J", best_braking_profile.regenerated/3600,"Wh")
#         print(distance_to_brake,"m Braking energy",best_braking_profile.braking_energy,"J", best_braking_profile.braking_energy/3600,"Wh")

#         printable_profile= best_braking_profile.build()
#         printable_profile = [str(p/V_mass)+" m/s2" if isinstance(p,float) else str(p.v*3.6)+"km/h:"+str(p.s) for p in printable_profile]
#         print(printable_profile)
#     else:
#         print("No profile match")
#     sys.stdout.flush()
