//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjeloles kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szoke Tibor Adam
// Neptun : GQ5E7S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 30;
float rnd() { return (float)rand() / RAND_MAX; }

//---------------------------
struct Camera {
	//---------------------------
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
	bool close = false;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.1f; bp = 10;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) *
			mat4(u.x, v.x, w.x, 0,
				u.y, v.y, w.y, 0,
				u.z, v.z, w.z, 0,
				0, 0, 0, 1);
	}
	mat4 P() {
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(vec3 i, vec3 j, vec3 pos) {
		if (close) {
			wEye = pos + j * 0.4f - i * 0.35f;
			wLookat = pos;
			wVup = i;
		}
		else {
			wEye = pos + j * 3.0f - i;
			wLookat = pos;
			wVup = i;
		}
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 vec(d2 * q1.w + d1 * q2.w + cross(d1, d2));
	return vec4(vec.x, vec.y, vec.z, q1.w * q2.w - dot(d1, d2));
}

vec3 Rotate(vec3 u, vec4 q) {
	vec4 qinv(-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
	return vec3(qr.x, qr.y, qr.z);
}

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		vec4 q(sinf(dt / 4.0f) * cosf(dt) / 2.0f, sinf(dt / 4.0f) * sinf(dt) / 2.0f, sinf(dt / 4.0f) * sqrtf(3.0f / 4.0f), cosf(dt / 4.0f));
		vec3 light3(wLightPos.x, wLightPos.y, wLightPos.z);
		light3 = Rotate(light3, q);
		wLightPos = vec4(light3.x, light3.y, light3.z, 0.0f);
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

//---------------------------
struct CheckerBoardTexture : public Texture {
	//---------------------------
	CheckerBoardTexture(vec3 color1, vec3 color2, const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? color1 : color2;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct LadyBugTex : public Texture {
	//---------------------------
	LadyBugTex(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(width * height);
		std::vector<vec2> dots(7);
		dots.push_back(vec2(width / 3.0f, height / 3.0f));
		dots.push_back(vec2(width / 3.0f * 2.0f, height / 4.0f * 3.0f));
		dots.push_back(vec2(width / 6.0f * 4.0f, height / 7.0f * 3.0f));
		dots.push_back(vec2(width / 5.0f * 3.0f, height / 7.0f));
		dots.push_back(vec2(width / 3.0f, height / 9.0f));
		dots.push_back(vec2(width / 2.0f, height / 2.0f));
		dots.push_back(vec2(width / 3.0f, height / 3.0f * 2.0f));
		const vec3 red(1, 0, 0);
		const vec3 black(0, 0, 0);
		float radius = width / 20.0f;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				vec2 pos(vec2((float)x, (float)y));
				for (vec2 dot : dots)
					if (length(dot - pos) < radius) {
						image[y * width + x] = black;
						break;
					}
					else
						image[y * width + x] = red;
			}
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//--------------------------
public:
	virtual void Bind(RenderState state) = 0;
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
 
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer
 
		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;
 
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye
 
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;
 
		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;
 
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer
 
		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.lights[0].wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

//---------------------------
class HalfEllipsoid : public ParamSurface {
	//---------------------------
	float a, b, c;

public:
	HalfEllipsoid() {
		a = 0.5f;
		b = 0.25f;
		c = 0.3f;
		Create();
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * (float)M_PI;
		float V = v * (float)M_PI;

		Clifford xU = Sin(T(U)) * a * Cos(V);
		Clifford yU = Sin(T(U)) * b * Sin(V);
		Clifford zU = Cos(T(U)) * c;
		vec3 drdU(xU.d, yU.d, zU.d);

		Clifford xV = Cos(T(V)) * a * Sin(U);
		Clifford yV = Sin(T(V)) * b * Sin(U);
		Clifford zV = 0;
		vec3 drdV(xV.d, yV.d, zV.d);

		vd.position = vec3(xU.f, yU.f, zU.f);
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Klein : public ParamSurface {
	//---------------------------
public:
	Klein() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2 * (float)M_PI;
		float V = v * 2 * (float)M_PI;
		bool UIsLargerThanPI = ((float)M_PI < U);

		Clifford aU = Cos(T(U)) * (Sin(T(U)) + 1) * 6;
		Clifford bU = Sin(T(U)) * 16;
		Clifford cU = Cos(T(U)) * -2 + 4;
		Clifford xU = UIsLargerThanPI ? aU + cU * cosf(V + (float)M_PI) : aU + cU * Cos(T(U)) * cosf(V);
		Clifford yU = UIsLargerThanPI ? bU : bU + cU * Sin(T(U)) * cosf(V);
		Clifford zU = cU * sinf(V);
		vec3 drdU(xU.d, yU.d, zU.d);

		Clifford aV = cosf(U) * (sinf(U) + 1) * 6;
		Clifford bV = sinf(U) * 16;
		Clifford cV = cosf(U) * -2 + 4;
		Clifford xV = UIsLargerThanPI ? aV + cV * Cos(T(V + (float)M_PI)) : aV + cV * cosf(U) * Cos(T(V));
		Clifford yV = UIsLargerThanPI ? bV : bV + cV * sinf(U) * Cos(T(V));
		Clifford zV = cV * Sin(T(V));
		vec3 drdV(xV.d, yV.d, zV.d);

		vd.position = vec3(xU.f, yU.f, zU.f);
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Dini : public ParamSurface {
	//---------------------------
	float a, b;
public:
	Dini() {
		a = 1.0f;
		b = 0.15f;
		Create();
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 4 * (float)M_PI;
		float V = v - (v - 1) * 0.01f;

		Clifford xU = Cos(T(U)) * a * Sin(V);
		Clifford yU = Sin(T(U)) * a * Sin(V);
		Clifford zU = (Cos(V) + Log(Tan(V / 2))) * a + T(U) * b;
		vec3 drdU(xU.d, yU.d, zU.d);

		Clifford xV = Cos(U) * a * Sin(T(V));
		Clifford yV = Sin(U) * a * Sin(T(V));
		Clifford zV = (Cos(T(V)) + Log(Tan(T(V) / 2))) * a + U * b;
		vec3 drdV(xV.d, yV.d, zV.d);

		vd.position = vec3(xU.f, yU.f, zU.f);
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	vec3 i, j, k, pos;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	vec3 geti() { return i; }
	vec3 getj() { return j; }
	vec3 getPos() { return pos; }

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { }
};

struct LadyBugObject : public Object {
	float u, v, velocity, alpha;
public:
	LadyBugObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry, float _u, float _v) : Object(_shader, _material, _texture, _geometry) {
		u = _u;
		v = _v;
		velocity = 1.0f;
		alpha = 0.0f;
	};

	void increaseAlpha() { alpha += (float)M_PI / 8; }
	void decreaseAlpha() { alpha -= (float)M_PI / 8; }

	void Draw(RenderState state) {
		mat4 T(i.x, i.y, i.z, 0,
			j.x, j.y, j.z, 0,
			k.x, k.y, k.z, 0,
			0, 0, 0, 1);
		mat4 Tinv(i.x, j.x, k.x, 0,
			i.y, j.y, k.y, 0,
			i.z, j.z, k.z, 0,
			0, 0, 0, 1);
		state.M = T * ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z)) * Tinv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void Animate(float tstart, float tend) {
		float U = u * 2 * (float)M_PI;
		float V = v * 2 * (float)M_PI;
		bool UIsLargerThanPI = ((float)M_PI < U);

		Clifford aU = Cos(T(U)) * (Sin(T(U)) + 1) * 6;
		Clifford bU = Sin(T(U)) * 16;
		Clifford cU = Cos(T(U)) * -2 + 4;
		Clifford xU = UIsLargerThanPI ? aU + cU * cosf(V + (float)M_PI) : aU + cU * Cos(T(U)) * cosf(V);
		Clifford yU = UIsLargerThanPI ? bU : bU + cU * Sin(T(U)) * cosf(V);
		Clifford zU = cU * sinf(V);
		vec3 drdU(xU.d, yU.d, zU.d);

		Clifford aV = cosf(U) * (sinf(U) + 1) * 6;
		Clifford bV = sinf(U) * 16;
		Clifford cV = cosf(U) * -2 + 4;
		Clifford xV = UIsLargerThanPI ? aV + cV * Cos(T(V + (float)M_PI)) : aV + cV * cosf(U) * Cos(T(V));
		Clifford yV = UIsLargerThanPI ? bV : bV + cV * sinf(U) * Cos(T(V));
		Clifford zV = cV * Sin(T(V));
		vec3 drdV(xV.d, yV.d, zV.d);

		pos = vec3(xU.f, yU.f, zU.f) * 0.1f;		// 0.1 = Klein scale
		vec3 normal = cross(drdU, drdV);

		translation = pos;
		i = normalize(drdU);
		j = normalize(normal);
		k = normalize(cross(i, j));

		rotationAxis = j;
		rotationAngle = alpha;

		float du = velocity * (tend - tstart) * cosf(alpha) / length(drdU);
		float dv = velocity * (tend - tstart) * sinf(alpha) / length(drdV);

		bool uIsLessThanMinimum = (u < -0.085f);
		bool uIsMoreThanMaximum = (1.0f < u);
		bool duPlus = du >= 0.0f ? true : false;

		if ((uIsLessThanMinimum && duPlus) || (uIsMoreThanMaximum && !duPlus) || (!uIsLessThanMinimum && !uIsMoreThanMaximum))
			u += du;
		// Teleportation
		if (uIsLessThanMinimum) {
			u = 0.637f;
			v = 0.5f;
			alpha = 0;
		}
		v += dv;
	}
};

struct FlowerObject : public Object {
	float u, v;
public:
	FlowerObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry, float _u, float _v) : Object(_shader, _material, _texture, _geometry) {
		u = _u;
		v = _v;
		place();
	};

	void Draw(RenderState state) {
		mat4 T(i.x, i.y, i.z, 0,
			j.x, j.y, j.z, 0,
			k.x, k.y, k.z, 0,
			0, 0, 0, 1);
		mat4 Tinv(i.x, j.x, k.x, 0,
			i.y, j.y, k.y, 0,
			i.z, j.z, k.z, 0,
			0, 0, 0, 1);
		state.M = T * ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z)) * Tinv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void place() {
		float U = u * 2 * (float)M_PI;
		float V = v * 2 * (float)M_PI;
		bool UIsLargerThanPI = ((float)M_PI < U);

		Clifford aU = Cos(T(U)) * (Sin(T(U)) + 1) * 6;
		Clifford bU = Sin(T(U)) * 16;
		Clifford cU = Cos(T(U)) * -2 + 4;
		Clifford xU = UIsLargerThanPI ? aU + cU * cosf(V + (float)M_PI) : aU + cU * Cos(T(U)) * cosf(V);
		Clifford yU = UIsLargerThanPI ? bU : bU + cU * Sin(T(U)) * cosf(V);
		Clifford zU = cU * sinf(V);
		vec3 drdU(xU.d, yU.d, zU.d);

		Clifford aV = cosf(U) * (sinf(U) + 1) * 6;
		Clifford bV = sinf(U) * 16;
		Clifford cV = cosf(U) * -2 + 4;
		Clifford xV = UIsLargerThanPI ? aV + cV * Cos(T(V + (float)M_PI)) : aV + cV * cosf(U) * Cos(T(V));
		Clifford yV = UIsLargerThanPI ? bV : bV + cV * sinf(U) * Cos(T(V));
		Clifford zV = cV * Sin(T(V));
		vec3 drdV(xV.d, yV.d, zV.d);

		pos = vec3(xU.f, yU.f, zU.f) * 0.1f;		// 0.1 = Klein scale
		vec3 normal = cross(drdU, drdV);

		j = normalize(drdU);
		k = normalize(normal);
		i = normalize(cross(j, k));
		translation = pos + k * 0.2f;
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object *> objects;

public:
	Camera camera;
	LadyBugObject * ladybugObject;
	std::vector<Light> lights;

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		Shader * nprShader = new NPRShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4.0f, 4.0f, 4.0f);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100.0f;

		Material * material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30.0f;

		// Textures
		Texture * texture5x10 = new CheckerBoardTexture(vec3(0, 0, 0), vec3(0, 1, 0), 5, 10);
		Texture * texture15x20 = new CheckerBoardTexture(vec3(1, 1, 0), vec3(0, 0, 0), 15, 20);
		Texture * ladybug = new LadyBugTex(500, 500);

		// Geometries
		Geometry * klein = new Klein();
		Geometry * dini = new Dini();
		Geometry * halfellipsoid = new HalfEllipsoid();

		// HalfEllipsoid
		ladybugObject = new LadyBugObject(nprShader, material0, ladybug, halfellipsoid, 0.5f, 0.25f);
		ladybugObject->scale = vec3(0.3f, 0.3f, 0.3f);
		objects.push_back(ladybugObject);

		for (int i = 0; i < 20; i++) {
			FlowerObject * diniObject = new FlowerObject(phongShader, material1, texture5x10, dini, rnd(), rnd());
			diniObject->scale = vec3(0.05f, 0.05f, 0.05f);
			objects.push_back(diniObject);
		}

		// Klein
		Object * kleinObject = new Object(phongShader, material0, texture15x20, klein);
		kleinObject->translation = vec3(0, 0, 0);
		kleinObject->rotationAxis = vec3(0, 1, 1);
		kleinObject->scale = vec3(0.1f, 0.1f, 0.1f);
		objects.push_back(kleinObject);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(5, 5, 4, 0);
		lights[0].La = vec3(0.5f, 0.5f, 0.5f);
		lights[0].Le = vec3(3, 3, 3);
	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects)
			obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera.Animate(objects.at(0)->geti(), objects.at(0)->getj(), objects.at(0)->getPos());
		lights.at(0).Animate(tstart, tend);
		objects.at(0)->Animate(tstart, tend);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case ' ':
		scene.camera.close = !scene.camera.close;
		break;
	case 'a':
		scene.ladybugObject->increaseAlpha();
		break;
	case 's':
		scene.ladybugObject->decreaseAlpha();
		break;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }
void onMouse(int button, int state, int pX, int pY) { }
void onMouseMotion(int pX, int pY) { }

void onIdle() {
	static float tend = 0.0f;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}