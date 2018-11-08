#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <sstream>

using namespace std;

const float pi = 3.141592653589793238462643383;

int bern = 0;
int transHThreads = 8;
int transHTrainTimes = 1000;
int nbatches = 100;
int dimension = 50;
float transHAlpha = 0.001;
float margin = 1;

string inPath = "../../";
string outPath = "../../";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

struct cmp_list {
	int minimal(int a,int b) {
		if (a > b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}
};

Triple *trainHead, *trainTail, *trainList;

/*
	There are some math functions for the program initialization.
*/

unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

float vec_len(float *con) {
	float res = 0;
	for (int i = 0; i < dimension; i++)
		res += con[i] * con[i];
	return sqrt(res);
}

void norm(float *con) {
	float x = vec_len(con);
	if (x > 1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm2one(float *con) {
	float x = vec_len(con);
	for (int i = 0; i < dimension; i++)
		con[i] /= x;
}


void norm(float *con, float *A) {
	norm2one(A);
	float x = 0;
	for (int i = 0; i < dimension; i++) {
		x += A[i] * con[i];
	}
	if (x > 0.1) {
		for (int i = 0; i < dimension; i++) {
			con[i] -= transHAlpha * A[i];
			A[i] -= transHAlpha * con[i];
		}
	}
	norm2one(A);
}

int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *A;
float *relationVecDao, *entityVecDao, *ADao;
float *tmpValue;

void norm(int h, int t, int r, int j) {
	norm(entityVecDao + h * dimension);
	norm(entityVecDao + t * dimension);
	norm(entityVecDao + j * dimension);
	norm(relationVecDao + r * dimension);
	norm2one(ADao + r * dimension);
	norm(entityVecDao + h * dimension, ADao + r * dimension);
	norm(entityVecDao + t * dimension, ADao + r * dimension);
	norm(entityVecDao + j * dimension, ADao + r * dimension);
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimension * 2 + entityTotal * dimension * 2 + relationTotal * dimension * 2, sizeof(float));
	relationVecDao = relationVec + relationTotal * dimension;
	entityVec = relationVecDao + relationTotal * dimension;
	entityVecDao = entityVec + entityTotal * dimension;
	A = entityVecDao + entityTotal * dimension;
	ADao = A + relationTotal * dimension;
	freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
	freqEnt = freqRel + relationTotal;
	
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i * dimension);
	}

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal * 3, sizeof(Triple));
	trainTail = trainHead + tripleTotal;
	trainList = trainTail + tripleTotal;
	for (int i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
		freqEnt[trainList[i].t]++;
		freqEnt[trainList[i].h]++;
		freqRel[trainList[i].r]++;
		trainHead[i] = trainList[i];
		trainTail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());
	sort(trainList, trainList + tripleTotal, cmp_list());

	lefHead = (int *)calloc(entityTotal * 4, sizeof(int));
	rigHead = lefHead + entityTotal;
	lefTail = rigHead + entityTotal;
	rigTail = lefTail + entityTotal;
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (float *)calloc(relationTotal * 2,sizeof(float));
	right_mean = left_mean + relationTotal;
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}

	for (int i = 0; i < relationTotal; i++) {
		for (int j = 0; j < dimension; j++)
			A[i * dimension + j] = randn(0, 1.0 / dimension, -1, 1);
		norm2one(A + i * dimension);
	}
}

/*
	Training process of transH.
*/

int transHLen;
int transHBatch;
float res;

float calc_sum(int e1, int e2, int rel, float* value, float& tmp1, float& tmp2) {
	int lasta1 = e1 * dimension;
	int lasta2 = e2 * dimension;
	int lastRel = rel * dimension;
	tmp1 = tmp2 = 0;
	for (int i = 0; i < dimension; i++) {
		tmp1 += A[lastRel + i] * entityVec[lasta1 + i];
		tmp2 += A[lastRel + i] * entityVec[lasta2 + i];
	}
	float sum = 0;
	for (int i = 0; i < dimension; i++) {
		value[i] = (entityVec[lasta2 + i] - tmp2 * A[lastRel + i]) - (entityVec[lasta1 + i] - tmp1 * A[lastRel + i]) - relationVec[lastRel + i];
		sum += fabs(value[i]);
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, float* value, float tmp1, float tmp2) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastRel = rel_a * dimension;
	float sum_x = 0, x;
	for (int i = 0; i < dimension; i++) {
		if (value[i] > 0) {
			x = belta * transHAlpha;
			sum_x += A[lastRel + i];
		}
		else {
			x = -belta * transHAlpha;
			sum_x -= A[lastRel + i];
		}
		relationVecDao[lastRel + i] -= x;
		entityVecDao[lasta1 + i] -= x;
		entityVecDao[lasta2 + i] += x;
		ADao[lastRel + i] += x * tmp1;
		ADao[lastRel + i] -= x * tmp2;
	}
	for (int i = 0; i < dimension; i++) {
		ADao[lastRel + i] += belta * transHAlpha * sum_x * entityVec[lasta1 + i];
		ADao[lastRel + i] -= belta * transHAlpha * sum_x * entityVec[lasta2 + i];
	}
}


void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *p, float *n) {
	float tmp1, tmp2, tmp3, tmp4;
	float sum1 = calc_sum(e1_a, e2_a, rel_a, p, tmp1, tmp2);
	float sum2 = calc_sum(e1_b, e2_b, rel_b, n, tmp3, tmp4);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, p, tmp1, tmp2);
    	gradient(e1_b, e2_b, rel_b, 1, n, tmp3, tmp4);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transHtrainMode(void *con) {
	int id, pr, i, j;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	float *positive = tmpValue + (id * 2) * dimension;
	float *negative = tmpValue + (id * 2 + 1) * dimension;
	for (int k = transHBatch / transHThreads; k >= 0; k--) {
		i = rand_max(id, transHLen);
		if (bern)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, positive, negative);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, positive, negative);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
	pthread_exit(NULL);
}

void* train_transH(void *con) {
	transHLen = tripleTotal;
	transHBatch = transHLen / nbatches;
	next_random = (unsigned long long *)calloc(transHThreads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(transHThreads * dimension * 2, sizeof(float));
	memcpy(relationVecDao, relationVec, dimension * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(ADao, A, dimension * relationTotal * sizeof(float));
	for (int epoch = 0; epoch < transHTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transHThreads * sizeof(pthread_t));
			for (long a = 0; a < transHThreads; a++)
				pthread_create(&pt[a], NULL, transHtrainMode,  (void*)a);
			for (long a = 0; a < transHThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimension * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(A, ADao, dimension * relationTotal * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transH.
*/

void out_transH() {
		stringstream ss;
		ss << dimension;
		string dim = ss.str();
	
		FILE* f2 = fopen((outPath + "TransH_relation2vec_" + dim + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "TransH_entity2vec_" + dim + ".vec").c_str(), "w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "TransH_A_" + dim + ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int j = 0; j < dimension; j++) {
					fprintf(f1, "%f\t", A[i * dimension + j]);
				fprintf(f1,"\n");
		}
		fclose(f1);
}

/*
	Main function
*/

int main() {
	init();
	train_transH(NULL);
	out_transH();
	return 0;
}
