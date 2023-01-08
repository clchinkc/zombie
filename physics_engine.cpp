// https://blog.winter.dev/2020/designing-a-physics-engine/
// https://blog.winter.dev/2020/epa-algorithm/
// https://blog.winter.dev/2020/gjk-algorithm/
// https://github.com/IainWinter/IwEngine


// dynamic


struct vector3 {
    float x, y, z;
};

struct quaternion {
    float x, y, z, w;
};

struct Object {
	vector3 Position; // struct with 3 floats for x, y, z or i + j + k
	vector3 Velocity;
	vector3 Force;
	float Mass;
};

/*
Here is an example of how you can define a Object class in Python that includes a vector3 struct with 3 floats for the x, y, and z components:

class vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class Object:    
    def __init__(self, position, velocity, force, mass):
        self.Position = position
        self.Velocity = velocity
        self.Force = force
        self.Mass = mass
*/



class PhysicsWorld {
private:
	std::vector<Object*> m_objects;
	vector3 m_gravity = vector3(0, -9.81f, 0);
 
public:
	void AddObject   (Object* object) { /* ... */ }
	void RemoveObject(Object* object) { /* ... */ }
 
	void Step(
		float dt)
	{
		for (Object* obj : m_objects) {
			obj->Force += obj->Mass * m_gravity; // apply a force
 
			obj->Velocity += obj->Force / obj->Mass * dt;
			obj->Position += obj->Velocity * dt;
 
			obj->Force = vector3(0, 0, 0); // reset net force at the end
		}
	}
};

/*
This PhysicsWorld class stores a list of Object objects and a gravity vector. It includes functions to add and remove objects, and a Step() function that updates the position and velocity of each object based on the forces acting on it. The Step() function applies the gravity force to each object and updates the position and velocity of the object using the Euler method.

class PhysicsWorld:
    def __init__(self):
        self.m_objects = []
        self.m_gravity = vector3(0., -9.81, 0.)
    
    def AddObject(self, object):
        self.m_objects.append(object)
    
    def RemoveObject(self, object):
        self.m_objects.remove(object)
    
    def Step(self, dt):
        for obj in self.m_objects:
            obj.Force += obj.Mass * self.m_gravity // apply a force
            
            obj.Velocity += obj.Force / obj.Mass * dt
            obj.Position += obj.Velocity * dt
            
            obj.Force = vector3(0., 0., 0.) // reset net force at the end
*/


// collision detection


struct CollisionPoints {
	vector3 A; // Furthest point of A into B
	vector3 B; // Furthest point of B into A
	vector3 Normal; // B – A normalized
	float Depth;    // Length of B – A
	bool HasCollision;
};
 
struct Transform { // Describes an objects location
	vector3 Position;
	vector3 Scale;
	quaternion Rotation;
};

/*
The CollisionPoints struct includes a vector3 struct for the A, B, and Normal points, as well as float variables for the Depth and a boolean variable for the HasCollision flag. The Transform struct also includes a vector3 struct for the Position and Scale, and a quaternion for the Rotation.

class CollisionPoints:
    def __init__(self, A, B, Normal, Depth, HasCollision):
        self.A = A
        self.B = B
        self.Normal = Normal
        self.Depth = Depth
        self.HasCollision = HasCollision

class Transform:
    def __init__(self, Position, Scale, Rotation):
        self.Position = Position
        self.Scale = Scale
        self.Rotation = Rotation

A = Vector3(1, 2, 3)
B = Vector3(4, 5, 6)
Normal = Vector3(0, 1, 0)
cp = CollisionPoints(A, B, Normal, 1.0, True)

position = Vector3(0, 0, 0)
scale = Vector3(1, 1, 1)
rotation = Quaternion(0, 0, 0, 1)

transform = Transform(position, scale, rotation)

*/



struct Collider {
	virtual CollisionPoints TestCollision(
		const Transform* transform,
		const Collider* collider,
		const Transform* colliderTransform) const = 0;
 
	virtual CollisionPoints TestCollision(
		const Transform* transform,
		const SphereCollider* sphere,
		const Transform* sphereTransform) const = 0;
 
	virtual CollisionPoints TestCollision(
		const Transform* transform,
		const PlaneCollider* plane,
		const Transform* planeTransform) const = 0;
};


/*
This Collider class is an abstract base class that defines three pure virtual functions for testing collisions with other colliders. These functions take as input the Transform objects of the colliders being tested and return a CollisionPoints object indicating the details of the collision.

You can define derived classes for specific types of colliders, such as SphereCollider and PlaneCollider, that override these virtual functions to provide the specific collision tests for those colliders.

class Collider:
    def TestCollision(self, transform, collider, colliderTransform):
        pass

    def TestCollision(self, transform, sphere, sphereTransform):
        pass

    def TestCollision(self, transform, plane, planeTransform):
        pass

*/


struct SphereCollider
	: Collider
{
	vector3 Center;
	float Radius;
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const Collider* collider,
		const Transform* colliderTransform) const override
	{
		return collider->TestCollision(colliderTransform, this, transform);
	}
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const SphereCollider* sphere,
		const Transform* sphereTransform) const override
	{
		return algo::FindSphereSphereCollisionPoints(
			this, transform, sphere, sphereTransform);
	}
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const PlaneCollider* plane,
		const Transform* planeTransform) const override
	{
		return algo::FindSpherePlaneCollisionPoints(
			this, transform, plane, planeTransform);
	}
};

/*
This SphereCollider class is a derived class of the Collider class, and it overrides the virtual functions defined in the Collider class to provide specific collision tests for sphere-sphere and sphere-plane collisions. It includes a vector3 struct for the Center point and a float for the Radius.

class SphereCollider(Collider):   
    def __init__(self, Center, Radius):
        self.Center = Center
        self.Radius = Radius
    
    def TestCollision(self, transform, collider, colliderTransform):
        return collider.TestCollision(colliderTransform, self, transform)
    
    def TestCollision(self, transform, sphere, sphereTransform):
        return algo.FindSphereSphereCollisionPoints(
            self, transform, sphere, sphereTransform)
    
    def TestCollision(self, transform, plane, planeTransform):
        return algo.FindSpherePlaneCollisionPoints(
            self, transform, plane, planeTransform)

*/



struct PlaneCollider
	: Collider
{
	vector3 Plane;
	float Distance;
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const Collider* collider,
		const Transform* colliderTransform) const override
	{
		return collider->TestCollision(colliderTransform, this, transform);
	}
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const SphereCollider* sphere,
		const Transform* sphereTransform) const override
	{
		// reuse sphere code
		return sphere->TestCollision(sphereTransform, this, transform);
	}
 
	CollisionPoints TestCollision(
		const Transform* transform,
		const PlaneCollider* plane,
		const Transform* planeTransform) const override
	{
		return {}; // No plane v plane
	}
};

/*
This PlaneCollider class is a derived class of the Collider class, and it overrides the virtual functions defined in the Collider class to provide specific collision tests for plane-sphere and plane-plane collisions. It includes a vector3 struct for the Plane and a float for the Distance.

class PlaneCollider(Collider):
    def __init__(self, Plane, Distance):
        super().__init__()
        self.Plane = Plane
        self.Distance = Distance
    
    def TestCollision(self, transform, collider, colliderTransform):
        return collider.TestCollision(colliderTransform, self, transform)
    
    def TestCollision(self, transform, sphere, sphereTransform):
        return sphere.TestCollision(sphereTransform, self, transform)
    
    def TestCollision(self, transform, plane, planeTransform):
        return None # No plane v plane

*/



namespace algo {
	CollisionPoints FindSphereSphereCollisionPoints(
		const SphereCollider* a, const Transform* ta,
		const SphereCollider* b, const Transform* tb);
 
 
	CollisionPoints FindSpherePlaneCollisionPoints(
		const SphereCollider* a, const Transform* ta,
		const PlaneCollider* b, const Transform* tb);
}

/*
The algo namespace includes two static methods for finding the collision points between a sphere and another sphere or a plane. These functions take as input the SphereCollider and PlaneCollider objects being tested, as well as their corresponding Transform objects, and return a CollisionPoints object indicating the details of the collision.

class algo:
    @staticmethod
    def FindSphereSphereCollisionPoints(a, ta, b, tb):
        pass
    
    @staticmethod
    def FindSpherePlaneCollisionPoints(a, ta, b, tb):
        pass

cp = algo.FindSphereSphereCollisionPoints(a, ta, b, tb)
cp = algo.FindSpherePlaneCollisionPoints(a, ta, b, tb)
*/



struct Object {
	float Mass;
	vector3 Velocity;
	vector3 Force;
 
	Collider* Collider;
	Transform* Transform;
};

/*
This Object class includes a vector3 struct for the Velocity and Force, as well as a float for the Mass, and pointers to a Collider and Transform object.

class Object:
    def __init__(self, Mass, Velocity, Force, Collider, Transform):
        self.Mass = Mass
        self.Velocity = Velocity
        self.Force = Force
        self.Collider = Collider
        self.Transform = Transform

mass = 1.0
velocity = Vector3(0, 0, 0)
force = Vector3(0, 0, 0)
collider = SphereCollider(Vector3(0, 0, 0), 1.0)
transform = Transform(Vector3(0, 0, 0), Vector3(1, 1, 1), 0)
obj = Object(mass, velocity, force, collider, transform)

*/



struct Collision {
	Object* ObjA;
	Object* ObjB;
	CollisionPoints Points;
};

/*
This Collision struct includes pointers to two Object objects, as well as a CollisionPoints object. It is used to store the details of a collision between two objects.

class Collision:
    def __init__(self, ObjA, ObjB, Points):
        self.ObjA = ObjA
        self.ObjB = ObjB
        self.Points = Points

collision = Collision(objA, objB, cp)
*/



class PhysicsWorld {
private:
	std::vector<Object*> m_objects;
	vector3 m_gravity = vector3(0, -9.81f, 0);
 
public:
	void AddObject   (Object* object) { /* ... */ }
	void RemoveObject(Object* object) { /* ... */ }
 
	void Step(
		float dt)
	{
		ResolveCollisions(dt);
 
		for (Object* obj : m_objects) { /* ... */ }
	}
 
	void ResolveCollisions(
		float dt)
	{
		std::vector<Collision> collisions;
		for (Object* a : m_objects) {
			for (Object* b : m_objects) {
				if (a == b) break;
 
				if (    !a->Collider
					|| !b->Collider)
				{
					continue;
				}
 
				CollisionPoints points = a->Collider->TestCollision(
					a->Transform,
					b->Collider,
					b->Transform);
 
				if (points.HasCollision) {
					collisions.emplace_back(a, b, points);
				}
			}
		}
 
		// Solve collisions
	}
};

/*
This PhysicsWorld class includes a list of pointers to Object objects, as well as a vector3 struct for the gravity vector. It has methods for adding and removing objects from the world, stepping the simulation forward by a given time step dt, and resolving collisions between objects.

class PhysicsWorld:
    def __init__(self):
        self.m_objects = []
        self.m_gravity = vector3(0, -9.81, 0)
    
    def AddObject(self, object):
        self.objects.append(object)
    
    def RemoveObject(self, object):
       self.objects.remove(object)
    
    def Step(self, dt):
        self.ResolveCollisions(dt)
        for obj in self.objects:
            # ...
    
    def ResolveCollisions(self, dt):
        collisions = []
        for a in self.m_objects:
            for b in self.m_objects:
                if a == b:
                    continue
                if not a.Collider or not b.Collider:
                    continue
                points = a.Collider.TestCollision(
                    a.Transform, b.Collider, b.Transform)
                if points.HasCollision:
                    collisions.append(Collision(a, b, points))

*/



CollisionPoints PlaneCollider::TestCollision(
	const Transform* transform,
	const SphereCollider* sphere,
	const Transform* sphereTransform) const
{
	// reuse sphere code
	CollisionPoints points = sphere->TestCollision(sphereTransform, this, transform);
 
	vector3 T = points.A; // You could have an algo Plane v Sphere to do the swap
	points.A = points.B;
	points.B = T;
 
	points.Normal = -points.Normal;
 
	return points;
}

/*
class PlaneCollider:
    def __init__(self, Plane, Distance):
        self.Plane = Plane
        self.Distance = Distance
    
    def TestCollision(self, transform, sphere, sphereTransform):
        # reuse sphere code
        points = sphere.TestCollision(sphereTransform, self, transform)
        
        T = points.A # You could have an algo Plane v Sphere to do the swap
        points.A = points.B
        points.B = .T
        
        points.Normal = -points.Normal
        
        return points

points = plane.TestCollision(transform, sphere, sphereTransform)
*/



// collision response



class Solver {
public:
	virtual void Solve(
		std::vector<Collision>& collisions,
		float dt) = 0;
};

/*
class Solver:
    def Solve(self, collisions, dt):
        pass


*/



class PhysicsWorld {
private:
	std::vector<Object*> m_objects;
	std::vector<Solver*> m_solvers;
	vector3 m_gravity = vector3(0, -9.81f, 0);
 
public:
	void AddObject   (Object* object) { /* ... */ }
	void RemoveObject(Object* object) { /* ... */ }
 
	void AddSolver   (Solver* solver) { /* ... */ }
	void RemoveSolver(Solver* solver) { /* ... */ }
 
	void Step(float dt) { /* ... */ }
 
	void ResolveCollisions(
		float dt)
	{
		std::vector<Collision> collisions;
		for (Object* a : m_objects) { /* ... */ }
 
		for (Solver* solver : m_solvers) {
			solver->Solve(collisions, dt);
		}
	}
};

/*
This PhysicsWorld class includes a list of pointers to Object objects, as well as a list of pointers to Solver objects.

class PhysicsWorld:
    def __init__(self):
        self.m_objects = []
        self.m_solvers = []
        self.m_gravity = Vector3(0, -9.81, 0)
    
    // adders and removers objects and solvers
    
    def Step(self, dt):
        self.ResolveCollisions(dt)
    
    def ResolveCollisions(self, dt):
        collisions = []
        for a in self.m_objects:
            for b in self.m_objects:
                if a == b:
                    continue
                if not a.Collider or not b.Collider:
                    continue
                points = a.Collider.TestCollision(
                    a.Transform, b.Collider, b.Transform)
                if points.HasCollision:
                    collisions.append(Collision(a, b, points))

                //if a.IsColliding(b):
                //    collisions.append(Collision(a, b))
        
        for solver in self.m_solvers:
            solver.Solve(collisions, dt)

*/



// more options



struct CollisionObject {
protected:
	Transform* m_transform;
	Collider* m_collider;
	bool m_isTrigger;
	bool m_isDynamic;
 
	std::function<void(Collision&, float)> m_onCollision;
 
public:
	// getters & setters, no setter for isDynamic
};

/*
This CollisionObject class has pointers to a Transform object and a Collider object, as well as boolean values for whether the object is a trigger and whether it is dynamic. It also has a function pointer to a callback function that will be called when the object collides with another object. The class includes getters and setters for all of these member variables, except for m_isDynamic, which can only be set in the constructor.

class CollisionObject(object):
	def __init__(self, transform, collider, isTrigger, isDynamic, onCollision):
		self.m_transform = transform
		self.m_collider = collider
		self.m_isTrigger = isTrigger
		self.m_isDynamic = isDynamic
        self.m_onCollision = lambda collision, dt: None // or None
*/



struct Rigidbody
	: CollisionObject
{
private:
	vector3 m_gravity;  // Gravitational acceleration
	vector3 m_force;    // Net force
	vector3 m_velocity;
 
	float m_mass;
	bool m_takesGravity; // If the rigidbody will take gravity from the world.
 
	float m_staticFriction;  // Static friction coefficient
	float m_dynamicFriction; // Dynamic friction coefficient
	float m_restitution;     // Elasticity of collisions (bounciness)
 
public:
	// getters & setters
};


/*
class Rigidbody(CollisionObject):
    def __init__(self, mass=1.0, takesGravity=True, staticFriction=1.0, dynamicFriction=1.0, restitution=0.5):
        super().__init__()
        self.m_gravity = vector3(0, 0, 0) # Gravitational acceleration
        self.m_force = vector3(0, 0, 0) # Net force
        self.m_velocity = vector3(0, 0, 0) 
        self.m_mass = mass
        self.m_takesGravity = takesGravity # if the rigidbody will take gravity from the world
        self.m_staticFriction = staticFriction # Static friction coefficient
        self.m_dynamicFriction = dynamicFriction # Dynamic friction coefficient
        self.m_restitution = restitution # Elasticity of collisions (bounciness)
    
    // getters & setters
*/



class CollisionWorld {
protected:
	std::vector<CollisionObject*> m_objects;
	std::vector<Solver*> m_solvers;
 
	std::function<void(Collision&, float)> m_onCollision;
 
public:
	void AddCollisionObject   (CollisionObject* object) { /* ... */ }
	void RemoveCollisionObject(CollisionObject* object) { /* ... */ }
 
	void AddSolver   (Solver* solver) { /* ... */ }
	void RemoveSolver(Solver* solver) { /* ... */ }
 
	void SetCollisionCallback(std::function<void(Collision&, float)>& callback) { /* ... */ }
 
	void SolveCollisions(
		std::vector<Collision>& collisions,
		float dt)
	{
		for (Solver* solver : m_solvers) {
			solver->Solve(collisions, dt);
		}
	}
 
	void SendCollisionCallbacks(
		std::vector<Collision>& collisions,
		float dt)
	{
		for (Collision& collision : collisions) {
			m_onCollision(collision, dt);
 
			auto& a = collision.ObjA->OnCollision();
			auto& b = collision.ObjB->OnCollision();
 
			if (a) a(collision, dt);
			if (b) b(collision, dt);
		}
	}
 
	void ResolveCollisions(
		float dt)
	{
		std::vector<Collision> collisions;
		std::vector<Collision> triggers;
		for (CollisionObject* a : m_objects) {
			for (CollisionObject* b : m_objects) {
				if (a == b) break;
 
				if (    !a->Col()
					|| !b->Col())
				{
					continue;
				}
 
				CollisionPoints points = a->Col()->TestCollision(
					a->Trans(),
					b->Col(),
					b->Trans());
 
				if (points.HasCollision) {
					if (    a->IsTrigger()
						|| b->IsTrigger())
					{
						triggers.emplace_back(a, b, points);
					}
 
					else {
						collisions.emplace_back(a, b, points);
					}
				}
			}
		}
 
		SolveCollisions(collisions, dt); // Don't solve triggers
 
		SendCollisionCallbacks(collisions, dt);
		SendCollisionCallbacks(triggers, dt);
	}
};

/*
class CollisionWorld:
    def init(self):
        self.m_objects = []
        self.m_solvers = []
        self.m_on_collision = None

    def add_collision_object(self, object):
        """Add a collision object to the world."""
        self.m_objects.append(object)

    def remove_collision_object(self, object):
        """Remove a collision object from the world."""
        self.m_objects.remove(object)

    def add_solver(self, solver):
        """Add a solver to the world."""
        self.m_solvers.append(solver)

    def remove_solver(self, solver):
        """Remove a solver from the world."""
        self.m_solvers.remove(solver)

    def set_collision_callback(self, callback):
        """Set the function to be called when a collision occurs."""
        self.m_on_collision = callback

    def solve_collisions(self, collisions, dt):
        """Solve the given collisions using the solvers in the world."""
        for solver in self.m_solvers:
            solver.solve(collisions, dt)

    def send_collision_callbacks(self, collisions, dt):
        """Send the collision callbacks for the given collisions."""
        for collision in collisions:
            if self.m_on_collision:
                self.m_on_collision(collision, dt)
            if collision.ObjA.on_collision:
                collision.ObjA.on_collision(collision, dt)
            if collision.ObjB.on_collision:
                collision.ObjB.on_collision(collision, dt)

    def resolve_collisions(self, dt):
        collisions = []
        triggers = []
        for a in m_objects:
            for b in m_objects:
                if a == b:
                    break
                if not a.col() or not b.col():
                    continue
                points = a.col().test_collision(a.trans(), b.col(), b.trans())
                if points.has_collision:
                    if a.is_trigger() or b.is_trigger():
                        triggers.append(Collision(a, b, points))
                    else:
                        collisions.append(Collision(a, b, points))
        self.solve_collisions(collisions, dt) # Don't solve triggers
        self.send_collision_callbacks(collisions, dt)
        self.send_collision_callbacks(triggers, dt)
*/



class DynamicsWorld
	: public CollisionWorld
{
private:
	vector3 m_gravity = vector3(0, -9.81f, 0);
 
public:
	void AddRigidbody(
		Rigidbody* rigidbody)
	{
		if (rigidbody->TakesGravity()) {
			rigidbody->SetGravity(m_gravity);
		}
 
		AddCollisionObject(rigidbody);
	}
 
	void ApplyGravity() {
		for (CollisionObject* object : m_objects) {
			if (!object->IsDynamic()) continue;
 
			Rigidbody* rigidbody = (Rigidbody*)object;
			rigidbody->ApplyForce(rigidbody->Gravity() * rigidbody->Mass());
		}
	}
 
	void MoveObjects(
		float dt)
	{
		for (CollisionObject* object : m_objects) {
			if (!object->IsDynamic()) continue;
 
			Rigidbody* rigidbody = (Rigidbody*)object;
 
			vector3 vel = rigidbody->Velocity()
					  + rigidbody->Force() / rigidbody->Mass()
					  * dt;
 
			rigidbody->SetVelocity(vel);

			vector3 pos = rigidbody->Position()
					  + rigidbody->Velocity()
					  * dt;
 
			rigidbody->SetPosition(pos);
 
			rigidbody->SetForce(vector3(0, 0, 0));
		}
	}
 
	void Step(
		float dt)
	{
		ApplyGravity();
		ResolveCollisions(dt);
		MoveObjects(dt);
	}
};

/*
class DynamicsWorld(CollisionWorld):
    def __init__(self):
        super().__init__()
        self.m_gravity = vector3(0, -9.81, 0)
    
    def add_rigidbody(self, rigidbody):
        if rigidbody.takes_gravity():
            rigidbody.set_gravity(self.m_gravity)
        
        self.add_collision_object(rigidbody)
    
    def apply_gravity(self):
        for object in self.m_objects:
            if not object.is_dynamic():
                continue
            
            rigidbody = Rigidbody(object)
            rigidbody.apply_force(rigidbody.gravity() * rigidbody.mass())
    
    def move_objects(self, dt):
        for object in self.m_objects:
            if not object.is_dynamic():
                continue
            
            rigidbody = Rigidbody(object)
            
            vel = rigidbody.velocity() + rigidbody.force() / rigidbody.mass() * dt
            
            rigidbody.set_velocity(vel)
            
            pos = rigidbody.position() + rigidbody.velocity() * dt
            
            rigidbody.set_position(pos)
            
            rigidbody.set_force(vector3(0, 0, 0))
    
    def step(self, dt):
        self.apply_gravity()
        self.resolve_collisions(dt)
        self.move_objects(dt)
*/



struct CollisionObject {
protected:
	Transform m_transform;
	Transform m_lastTransform;
	Collider* m_collider;
	bool m_isTrigger;
	bool m_isStatic;
	bool m_isDynamic;
 
	std::function<void(Collision&, float)> m_onCollision;
public:
	// Getters & setters for everything, no setter for isDynamic
};

/*
class CollisionObject:
    def init(self):
        self.m_transform = Transform()
        self.m_lastTransform = Transform()
        self.m_collider = Collider()
        self.m_isTrigger = False
        self.m_isStatic = False
        self.m_isDynamic = False
        self.m_onCollision = lambda collision, dt: None
        # getters & setters for everything, no setter for isDynamic
*/



class PhysicsSmoothStepSystem {
private:
	float accumulator = 0.0f;
 
public:
	void Update() {
		for (Entity entity : GetAllPhysicsEntities()) {
			Transform*       transform = entity.Get<Transform>();
			CollisionObject* object    = entity.Get<CollisionObject>();
 
			Transform& last    = object->LastTransform();
			Transform& current = object->Transform();
 
			transform->Position = lerp(
				last.Position,
				current.Position,
				accumulator / PhysicsUpdateRate()
			);
		}
 
		accumulator += FrameDeltaTime();
	}
 
	void PhysicsUpdate() {
		accumulator = 0.0f;
	}
};

/*
class PhysicsSmoothStepSystem:
    def init(self):
        self.accumulator = 0.0

    def update(self):
        for entity in get_all_physics_entities():
            transform = entity.get(Transform)
            object = entity.get(CollisionObject)
            
            last_transform = object.last_transform()
            current_transform = object.transform()
            
            transform.position = lerp(
                last_transform.position,
                current_transform.position,
                self.accumulator / physics_update_rate()
            )
        
        self.accumulator += frame_delta_time()

    def physics_update(self):
        self.accumulator = 0.0

*/

