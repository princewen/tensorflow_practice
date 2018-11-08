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
int transDThreads = 8;
int transDTrainTimes = 1000;
int nbatches = 100;
int dimension = 50;
int dimensionR = 50;
float transDAlpha = 0.001;
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
		if (a < b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) < minimal(b.h, b.t));
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

void norm(float *con, int dimension) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(float *con, float *conTrans, float *relTrans) {
	float x = 0, ee = 0, tmp1 = 0;
	for (int i = 0; i < dimension; i++)
		ee += con[i] * conTrans[i];
	for (int i = 0; i < dimensionR; i++) {
		float tmp = ee * relTrans[i];
		if (i < dimension)
			tmp = tmp + con[i];
		x += tmp * tmp;
		tmp1 += tmp * 2 * relTrans[i];
	}
	if (x > 1) {
		float lambda = 1;
		for (int i = 0; i < dimensionR; i++) {
			float tmp = ee * relTrans[i];
			if (i < dimension)
				tmp = tmp + con[i];
			tmp = tmp + tmp;
			relTrans[i] -= transDAlpha * lambda * tmp * ee;
			if (i < dimension)
				con[i] -= transDAlpha * lambda * tmp;
		}
		for (int i = 0; i < dimension; i++) {
			con[i] -= transDAlpha * lambda * tmp1 * conTrans[i];
			conTrans[i] -= transDAlpha * lambda * tmp1 * con[i];
		}
	}
}

int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *matrix;
float *relationVecDao, *entityVecDao, *matrixDao;
float *entityTransVec, *entityTransVecDao, *relationTransVec, *relationTransVecDao;
float *tmpValue;

void norm(int h, int t, int r, int j) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, entityTransVecDao + dimension * h, relationTransVecDao + dimensionR * r);
		norm(entityVecDao + dimension * t, entityTransVecDao + dimension * t, relationTransVecDao + dimensionR * r);
		norm(entityVecDao + dimension * j, entityTransVecDao + dimension * j, relationTransVecDao + dimensionR * r);
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

	relationVec = (float *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimensionR * 2 + entityTotal * dimension * 2, sizeof(float));
	relationVecDao = relationVec + relationTotal * dimensionR;
	entityVec = relationVecDao + relationTotal * dimensionR;
	entityVecDao = entityVec + entityTotal * dimension;
	relationTransVec = entityVecDao + entityTotal * dimension;
	relationTransVecDao = relationTransVec + relationTotal * dimensionR;
	entityTransVec = relationTransVecDao + relationTotal * dimensionR;
	entityTransVecDao = entityTransVec + entityTotal * dimension;

	freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
	freqEnt = freqRel + relationTotal;

	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii < dimensionR; ii++) {
			relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
			relationTransVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
		}
		norm(relationTransVec + i * dimensionR, dimensionR);
	}
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii < dimension; ii++) {
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			entityTransVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		}
		norm(entityVec + i * dimension, dimension);
		norm(entityTransVec + i * dimension, dimension);
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
}

/*
	Training process of transD.
*/

int transDLen;
int transDBatch;
float res;

float calc_sum(int e1, int e2, int rel, float *tmp1, float *tmp2, float &ee1, float &ee2) {
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimensionR;
	float sum = 0;
	ee1 = 0, ee2 = 0;
	for (int ii = 0; ii < dimension; ii++) {
		ee1 += entityTransVec[last1 + ii] * entityVec[last1 + ii];
		ee2 += entityTransVec[last2 + ii] * entityVec[last2 + ii];
	}
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp1[ii] = ee1 * relationTransVec[lastr + ii];
		tmp2[ii] = ee2 * relationTransVec[lastr + ii];
		if (ii < dimension) {
			tmp1[ii] += entityVec[last1 + ii];
			tmp2[ii] += entityVec[last2 + ii];
		}
		sum += fabs(tmp1[ii] + relationVec[lastr + ii] - tmp2[ii]);
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, int same, float *tmp1, float *tmp2, float &e1, float &e2) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	float x, s = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * transDAlpha;
		else
			x = -belta * transDAlpha;
		s += x * relationTransVec[lastar + ii];
		relationTransVecDao[lastar + ii] -= same * x * e1;
		relationTransVecDao[lastar + ii] += same * x * e2;
		relationVecDao[lastar + ii] -= same * x;
		if (ii < dimension) {
			entityVecDao[lasta1 + ii] -= same * x;
			entityVecDao[lasta2 + ii] += same * x;
		}
	}
	s = s * same;
	for (int ii = 0; ii < dimension; ii++) {
		entityVecDao[lasta1 + ii] -= s * entityTransVec[lasta1 + ii];
		entityTransVecDao[lasta1 + ii] -= s * entityVec[lasta1 + ii];
		entityVecDao[lasta2 + ii] += s * entityTransVec[lasta2 + ii];
		entityTransVecDao[lasta2 + ii] += s * entityVec[lasta2 + ii];
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *tmp) {
	float e1, e2, e3, e4;
	float sum1 = calc_sum(e1_a, e2_a, rel_a, tmp, tmp + dimensionR, e1, e2);
	float sum2 = calc_sum(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3, e3, e4);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR, e1, e2);
    	gradient(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3, e3, e4);
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

void* transDtrainMode(void *con) {
	int id, i, j, pr;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	float *tmp = tmpValue + id * dimensionR * 4;
	for (int k = transDBatch / transDThreads; k >= 0; k--) {
		i = rand_max(id, transDLen);	
		if (bern)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
	pthread_exit(NULL);
}

void* train_transD(void *con) {
	transDLen = tripleTotal;
	transDBatch = transDLen / nbatches;
	next_random = (unsigned long long *)calloc(transDThreads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(transDThreads * dimensionR * 4, sizeof(float));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(relationTransVecDao, relationTransVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityTransVecDao, entityTransVec, dimension * entityTotal * sizeof(float));
	for (int epoch = 0; epoch < transDTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transDThreads * sizeof(pthread_t));
			for (long a = 0; a < transDThreads; a++)
				pthread_create(&pt[a], NULL, transDtrainMode,  (void*)a);
			for (long a = 0; a < transDThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(relationTransVec, relationTransVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityTransVec, entityTransVecDao, dimension * entityTotal * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}

}

/*
	Get the results of transD.
*/

void out_transD() {
		stringstream ss;
		ss << dimension;
		string dim = ss.str();
	
		FILE* f2 = fopen((outPath + "TransD_relation2vec_" + dim + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "TransD_entity2vec_" + dim + ".vec").c_str(), "w");
		for (int i=0; i < relationTotal; i++) {
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
		FILE* f1 = fopen((outPath + "TransD_A_" + dim + ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimensionR * i;
			for (int ii = 0; ii < dimensionR; ii++)
				fprintf(f1, "%.6f\t", relationTransVec[last + ii]);
			fprintf(f1,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f1, "%.6f\t", entityTransVec[last + ii] );
			fprintf(f1,"\n");
		}
		fclose(f1);
}

/*
	Main function
*/

int main() {
	init();
	train_transD(NULL);
	out_transD();
	return 0;
}
