// https://blog.winter.dev/2020/designing-a-physics-engine/
// https://blog.winter.dev/2020/epa-algorithm/
// https://blog.winter.dev/2020/gjk-algorithm/
// https://github.com/IainWinter/IwEngine
// https://github.com/IainWinter/WinterFramework
// https://github.com/KrishnenduMarathe/physicsSimulator

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


namespace algo {
	CollisionPoints FindSphereSphereCollisionPoints(
		const SphereCollider* a, const Transform* ta,
		const SphereCollider* b, const Transform* tb);
 
 
	CollisionPoints FindSpherePlaneCollisionPoints(
		const SphereCollider* a, const Transform* ta,
		const PlaneCollider* b, const Transform* tb);
}


struct Object {
	float Mass;
	vector3 Velocity;
	vector3 Force;
 
	Collider* Collider;
	Transform* Transform;
};


struct Collision {
	Object* ObjA;
	Object* ObjB;
	CollisionPoints Points;
};


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



// collision response



class Solver {
public:
	virtual void Solve(
		std::vector<Collision>& collisions,
		float dt) = 0;
};



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



