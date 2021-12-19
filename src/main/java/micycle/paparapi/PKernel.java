package micycle.paparapi;

import com.aparapi.Kernel;
import com.aparapi.Range;

import processing.core.PApplet;

/**
 * A <i>PKernel</i> encapsulates a pixels parallel algorithm that will execute
 * either on a GPU (through conversion to OpenCL) or on a CPU via a Java Thread
 * Pool.
 * <p>
 * To write a new kernel, a developer extends the <code>Kernel</code> class and
 * overrides the <code>Kernel.run()</code> method. To execute this kernel, the
 * developer creates a new instance of it and calls
 * <code>Kernel.execute(int globalSize)</code> with a suitable 'global size'. At
 * runtime Aparapi will attempt to convert the <code>Kernel.run()</code> method
 * (and any method called directly or indirectly by <code>Kernel.run()</code>)
 * into OpenCL for execution on GPU devices made available via the OpenCL
 * platform.
 * 
 * @author Michael Carleton
 *
 */
public abstract class PKernel extends Kernel {

	protected final int width, height;
	protected final PApplet p;

	protected int[] pixels;

	protected final int[] random; // cannot be private
	protected final int seed; // cannot be private
	private float fps = 0;

	protected PKernel(PApplet p, int kernelSize) {
		this.p = p;
		width = p.width;
		height = p.height;
		p.loadPixels();
		pixels = new int[p.pixels.length];

		random = new int[kernelSize];
		for (int i = 0; i < kernelSize; i++) {
			random[i] = wang_hash(i);
		}
		seed = (int) (System.currentTimeMillis() % Integer.MAX_VALUE);
	}

	@Override
	public synchronized Kernel execute(int n) {
		long startTime = System.nanoTime();
		Kernel k = super.execute(Range.create(n));
		long endTime = System.nanoTime() - startTime;
		fps = (endTime) / 1000000f;
		fps = 1000 / fps;
		return k;
	}

	/**
	 * Dumps the kernel's pixel pixels back to Processing's pixels[].
	 */
	public void dump() {
		System.arraycopy(pixels, 0, p.pixels, 0, pixels.length);
	}

	/**
	 * Clear this kernel's pixel array.
	 */
	public void clear() {
		pixels = new int[p.pixels.length];
	}

	/**
	 * Get time of last call to {@link #execute(int)}.
	 * 
	 * @return
	 */
	public float getLastFPS() {
		return fps;
	}

	/**
	 * Converts a 1D array index into its 2D index/coordinate.
	 * 
	 * @param globalID
	 * @return
	 */
	protected int[] getIndex(int globalID) {
		return new int[] { globalID % width, globalID / height };
	}
	
	/**
	 * @param red   ∈[0, 255]
	 * @param green ∈[0, 255]
	 * @param blue  ∈[0, 255]
	 * @return
	 */
	protected static int composeColor(final int red, final int green, final int blue) {
		return -16777216 | red << 16 | green << 8 | blue;
	}

	/**
	 * 
	 * @param r ∈[0, 1]
	 * @param g ∈[0, 1]
	 * @param b ∈[0, 1]
	 * @param a
	 * @return
	 */
	protected int composeColor(float r, float g, float b, float a) {
		return (int) (a * 255) << 24 | ((int) (r * 255) << 16 | (int) (g * 255) << 8 | (int) (b * 255));
	}

	protected float map(float value, float min1, float max1, float min2, float max2) {
		return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
	}

	/**
	 * Gets the next value in the PRNG sequence for the given GID. Random hash
	 * (using xor-shift)
	 * 
	 * @param seed
	 * @return 0...1
	 */
	protected float randomHash(final int GID) {
		random[GID] ^= (random[GID] << 13);
		random[GID] ^= (random[GID] >> 17);
		random[GID] ^= (random[GID] << 5);
		return random[GID] * (1.0f / Integer.MAX_VALUE);
	}

	private static int wang_hash(int seed) {
		seed = (seed ^ 61) ^ (seed >> 16);
		seed *= 9;
		seed = seed ^ (seed >> 4);
		seed *= 0x27d4eb2d;
		seed = seed ^ (seed >> 15);
		return seed;
	}

	// GLSL-like Functions (aparapi doesn't support static methods)

	/**
	 * GLSL distance()
	 * 
	 * @param p1x
	 * @param p1y
	 * @param p2x
	 * @param p2y
	 * @return the distance between the two points p1 and p2
	 */
	protected float distance(float p1x, float p1y, float p2x, float p2y) {
		float deltaX = p2x - p1x;
		float deltaY = p2y - p1y;
		return sqrt(deltaX * deltaX + deltaY * deltaY);
	}

	/**
	 * glsl fract()
	 * 
	 * @param x the value to evaluate
	 * @return the fractional part of x. This is calculated as x - floor(x).
	 */
	protected float fract(float x) {
		return x - (int) x;
	}

	// PerlinNoiseLite Routine //

	// Hashing
	private static final int PrimeX = 501125321;
	private static final int PrimeY = 1136930381;
	private static final int PrimeZ = 1720413743;

	protected float noise(float x, float y) {
		int x0 = FastFloor(x);
		int y0 = FastFloor(y);

		float xd0 = (x - x0);
		float yd0 = (y - y0);
		float xd1 = xd0 - 1;
		float yd1 = yd0 - 1;

		float xs = InterpQuintic(xd0);
		float ys = InterpQuintic(yd0);

		x0 *= PrimeX;
		y0 *= PrimeY;
		int x1 = x0 + PrimeX;
		int y1 = y0 + PrimeY;

		float xf0 = Lerp(GradCoord(seed, x0, y0, xd0, yd0), GradCoord(seed, x1, y0, xd1, yd0), xs);
		float xf1 = Lerp(GradCoord(seed, x0, y1, xd0, yd1), GradCoord(seed, x1, y1, xd1, yd1), xs);

		return Lerp(xf0, xf1, ys) * 1.4247691104677813f;
	}

	protected float noise(float x, float y, float z) {
		int x0 = FastFloor(x);
		int y0 = FastFloor(y);
		int z0 = FastFloor(z);

		float xd0 = (x - x0);
		float yd0 = (y - y0);
		float zd0 = (z - z0);
		float xd1 = xd0 - 1;
		float yd1 = yd0 - 1;
		float zd1 = zd0 - 1;

		float xs = InterpQuintic(xd0);
		float ys = InterpQuintic(yd0);
		float zs = InterpQuintic(zd0);
		x0 *= PrimeX;
		y0 *= PrimeY;
		z0 *= PrimeZ;
		int x1 = x0 + PrimeX;
		int y1 = y0 + PrimeY;
		int z1 = z0 + PrimeZ;

		float xf00 = Lerp(GradCoord(seed, x0, y0, z0, xd0, yd0, zd0), GradCoord(seed, x1, y0, z0, xd1, yd0, zd0), xs);
		float xf10 = Lerp(GradCoord(seed, x0, y1, z0, xd0, yd1, zd0), GradCoord(seed, x1, y1, z0, xd1, yd1, zd0), xs);
		float xf01 = Lerp(GradCoord(seed, x0, y0, z1, xd0, yd0, zd1), GradCoord(seed, x1, y0, z1, xd1, yd0, zd1), xs);
		float xf11 = Lerp(GradCoord(seed, x0, y1, z1, xd0, yd1, zd1), GradCoord(seed, x1, y1, z1, xd1, yd1, zd1), xs);

		float yf0 = Lerp(xf00, xf10, ys);
		float yf1 = Lerp(xf01, xf11, ys);

		return Lerp(yf0, yf1, zs) * 0.964921414852142333984375f;
	}

	private float GradCoord(int seed, int xPrimed, int yPrimed, float xd, float yd) {
		int hash = Hash(seed, xPrimed, yPrimed);
		hash ^= hash >> 15;
		hash &= 127 << 1;

		float xg = Gradients2D_$constant$[hash];
		float yg = Gradients2D_$constant$[hash | 1];

		return xd * xg + yd * yg;
	}

	private float GradCoord(int seed, int xPrimed, int yPrimed, int zPrimed, float xd, float yd, float zd) {
		int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
		hash ^= hash >> 15;
		hash &= 63 << 2;

		float xg = Gradients3D_$constant$[hash];
		float yg = Gradients3D_$constant$[hash | 1];
		float zg = Gradients3D_$constant$[hash | 2];

		return xd * xg + yd * yg + zd * zg;
	}

	private static int Hash(int seed, int xPrimed, int yPrimed) {
		int hash = seed ^ xPrimed ^ yPrimed;
		hash *= 0x27d4eb2d;
		return hash;
	}

	private static int Hash(int seed, int xPrimed, int yPrimed, int zPrimed) {
		int hash = seed ^ xPrimed ^ yPrimed ^ zPrimed;
		hash *= 0x27d4eb2d;
		return hash;
	}

	private static float InterpQuintic(float t) {
		return t * t * t * (t * (t * 6 - 15) + 10);
	}

	private static int FastFloor(float f) {
		return f >= 0 ? (int) f : (int) f - 1;
	}

	private static float Lerp(float a, float b, float t) {
		return a + t * (b - a);
	}

	protected final float[] Gradients2D_$constant$ = { 0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f,
			0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f, 0.923879532511287f, 0.38268343236509f,
			0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f,
			0.130526192220052f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f,
			-0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f, -0.923879532511287f, -0.38268343236509f,
			-0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f,
			-0.130526192220052f, 0.99144486137381f, 0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f,
			0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f, 0.923879532511287f, 0.38268343236509f,
			0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f,
			0.130526192220052f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f,
			-0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f, -0.923879532511287f, -0.38268343236509f,
			-0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f,
			-0.130526192220052f, 0.99144486137381f, 0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f,
			0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f, 0.923879532511287f, 0.38268343236509f,
			0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f,
			0.130526192220052f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f,
			-0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f, -0.923879532511287f, -0.38268343236509f,
			-0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f,
			-0.130526192220052f, 0.99144486137381f, 0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f,
			0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f, 0.923879532511287f, 0.38268343236509f,
			0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f,
			0.130526192220052f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f,
			-0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f, -0.923879532511287f, -0.38268343236509f,
			-0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f,
			-0.130526192220052f, 0.99144486137381f, 0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f,
			0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f, 0.923879532511287f, 0.38268343236509f,
			0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f,
			0.130526192220052f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f,
			-0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f, -0.923879532511287f, -0.38268343236509f,
			-0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f,
			-0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.923879532511287f, 0.38268343236509f,
			0.923879532511287f, -0.38268343236509f, 0.38268343236509f, -0.923879532511287f, -0.38268343236509f, -0.923879532511287f,
			-0.923879532511287f, -0.38268343236509f, -0.923879532511287f, 0.38268343236509f, -0.38268343236509f, 0.923879532511287f, };

	protected float[] Gradients3D_$constant$ = { 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0,
			-1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1,
			0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1,
			-1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0,
			1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1,
			-1, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0,
			1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, 0, -1, -1, 0 };
}
