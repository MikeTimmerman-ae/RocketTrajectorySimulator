# Rocket Trajectory Simulator
6-DOF sounding rocket trajectory simulator

The trajectory of the rocket is obtained through solving a 
set of four ordinary differential equations describing the dynamics
of the rocket using a numerical solving technique, namely the
Runge-Kutta 45 method. 

The simulator comprises one main class, "trajectory", which has all
the rocket parameters stored and keeps track of the state variables
as the system of ODEs are solved. Following parameters can be specified
to describe a given rocket:
- empty mass
- body radius
- body length
- initial position of center of mass
- final position of center of mass
- position of center of pressure
- drag coefficient
- launch angle
- rocket engine (imported from www.thrustcurve.org)

This class also has a method which plots the final trajectory as
altitude-horizontal displacement and altitude-time.