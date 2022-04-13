# TO-DO: Change repository name to 'space-rocket-controller-clc-nonlinear-equations-of-motion' (?) 
# TO-DO: Containerize to microservice
import numpy as np

# Solves for the motion of the flight body
# Requires: Net external forces and moments (engine, air, and gravity)
# Required by: None
# Outputs: Acceleration, angular acceleration, ECI velocity, quaternion dot
# Used for time integration to calculate next time step

class NonlinearMotionSolver():
  def __init__(self, metadata_distributor):
    self.metadata_distributor = metadata_distributor
    self.xyz = self.metadata_distributor.get_var('xyz')
    self.uvw = self.metadata_distributor.get_var('uvw')
    self.pqr = self.metadata_distributor.get_var('pqr')
    self.lmn = self.metadata_distributor.get_var('lmn')
    self.q = self.metadata_distributor.get_var('q')
    self.t_hb = self.metadata_distributor.get_var('t_hb')
    self.flightbody_m = self.metadata_distributor.get_var('flightbody_m')
    self.flightbody_I = self.metadata_distributor.get_var('flightbody_I')

  def get_flight_body_acceleration(self):
    x, y, z = self.xyz
    u, v, w = self.uvw
    p, q, r = self.pqr
    u_dot = x/self.flightbody_m - q*w - r*v
    v_dot = y/self.flightbody_m - r*u - p*w
    w_dot = z/self.flightbody_m - p*v - q*u
    uvw_dot = np.array([u_dot, v_dot, w_dot])
    self.metadata_distributor.set({'uvw_dot': uvw_dot})
    return uvw_dot
  
  def get_flight_body_angular_acceleration(self):
    l, m, n = self.lmn
    p, q, r = self.pqr
    p_dot = (self.flightbody_I[2][2]*l + self.flightbody_I[2][0]*n 
            -(self.flightbody_I[2][2]**2 - self.flightbody_I[1][1]*self.flightbody_I[2][2] + self.flightbody_I[2][0]**2)*q*r 
            + self.flightbody_I[2][0]*(self.flightbody_I[0][0] - self.flightbody_I[1][1] + self.flightbody_I[2][2])*p*q) \
              / (self.flightbody_I[0][0]*self.flightbody_I[2][2]-self.flightbody_I[2][0]**2)
    q_dot = ((m + self.flightbody_I[2][2]-self.flightbody_I[0][0])*r*p
            + self.flightbody_I[2][0]*(r**2-p**2))/ self.flightbody_I[1][1]
    r_dot = (self.flightbody_I[2][0]*l + self.flightbody_I[0][0]*n
            + (self.flightbody_I[0][0]**2 - self.flightbody_I[0][0]*self.flightbody_I[1][1] + self.flightbody_I[2][0]**2)*p*q
            - self.flightbody_I[2][0]*(self.flightbody_I[0][0] - self.flightbody_I[1][1] + self.flightbody_I[2][2])*q*r) \
              / (self.flightbody_I[0][0]*self.flightbody_I[2][2]-self.flightbody_I[2][0]**2)
    pqr_dot = np.array([p_dot, q_dot, r_dot])
    self.metadata_distributor.set({'pqr_dot': pqr_dot})
    return pqr_dot

  def get_eci_dot(self):
    xyz_dot = np.matmul(self.t_hb.T, self.uvw)
    self.metadata_distributor.set({'xyz_dot': xyz_dot})
    return xyz_dot
  
  def get_qs_dot(self):
    p, q, r = self.pqr
    asym_pqr = np.array([[0,r,-q,p],[-r,0,p,q],[q,-p,0,r],[-p,-q,-r,0]])
    qs_dot = 0.5*np.matmul(asym_pqr,self.qs)
    self.metadata_distributor.set({'qs_dot': qs_dot})
    return qs_dot