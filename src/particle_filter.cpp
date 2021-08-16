/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#define pb push_back
using std::string;
using std::vector;
using namespace std;
static default_random_engine gen;
int i;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/**
	 * TODO: Set the number of particles. Initialize all particles to
	 *   first position (based on estimates of x, y, theta and their uncertainties
	 *   from GPS) and all weights to 1.
	 * TODO: Add random Gaussian noise to each particle.
	 * NOTE: Consult particle_filter.h for more information about this method
	 *   (and others in this file).
	 */
	//num_particles = 0;  // TODO: Set the number of particles

	if (is_initialized) {
		return;
	}

	num_particles = 100;

	normal_distribution<double> dx(x, std[0]);
	normal_distribution<double> dy(y, std[1]);
	normal_distribution<double> dt(theta, std[2]);

	for (i = 0; i < num_particles; i++) {

		Particle p;
		p.id = i;
		p.x = dx(gen);
		p.y = dy(gen);
		p.theta = dt(gen);
		p.weight = 1.0;
		particles.pb(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
	/**
	 * TODO: Add measurements to each particle and add random Gaussian noise.
	 * NOTE: When adding noise you may find std::normal_distribution
	 *   and std::default_random_engine useful.
	 *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	 *  http://www.cplusplus.com/reference/random/default_random_engine/
	 */
	normal_distribution<double> dx(0, std_pos[0]);
	normal_distribution<double> dy(0, std_pos[1]);
	normal_distribution<double> dt(0, std_pos[2]);

	for (i = 0; i < num_particles; i++) {

		if (yaw_rate > 0.00001 or yaw_rate < -0.00001) {

			double temp = velocity / yaw_rate;
			double t = yaw_rate * delta_t;
			particles[i].x += (temp * (sin(particles[i].theta + t) - sin(particles[i].theta)));
			particles[i].y += (temp * (cos(particles[i].theta) - cos(particles[i].theta + t)));
			particles[i].theta += t;
		}
		else {

			double temp = velocity * delta_t;
			particles[i].x += (temp * cos(particles[i].theta));
			particles[i].y += (temp * sin(particles[i].theta));
		}

		particles[i].x += dx(gen);
		particles[i].y += dy(gen);
		particles[i].theta += dt(gen);
	}

}

/*--------------------------------------  HAMDAN'S WORK DONE  -----------------------------------------*/

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
	/**
	 * TODO: Find the predicted measurement that is closest to each
	 *   observed measurement and assign the observed measurement to this
	 *   particular landmark.
	 * NOTE: this method will NOT be called by the grading code. But you will
	 *   probably find it useful to implement this method and use it as a helper
	 *   during the updateWeights phase.
	 */
	unsigned int obs = observations.size();
	unsigned int prd = predicted.size();
	for (unsigned int i = 0; i < obs; i++) { // For each observation
		double min_Distance = numeric_limits<double>::max();
		int index_mapId = -1;
		for (unsigned j = 0; j < prd; j++ ) { // For each predition.
			double x_Dist = observations[i].x - predicted[j].x;
			double y_Dist = observations[i].y - predicted[j].y;
			double distance = x_Dist * x_Dist + y_Dist * y_Dist;
			if (distance < min_Distance) {
				min_Distance = distance;
				index_mapId = predicted[j].id;
			}
			observations[i].id = index_mapId;
		}
	}
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
	/**
	 * TODO: Update the weights of each particle using a mult-variate Gaussian
	 *   distribution. You can read more about this distribution here:
	 *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	 * NOTE: The observations are given in the VEHICLE'S coordinate system.
	 *   Your particles are located according to the MAP'S coordinate system.
	 *   You will need to transform between the two systems. Keep in mind that
	 *   this transformation requires both rotation AND translation (but no scaling).
	 *   The following is a good resource for the theory:
	 *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	 *   and the following is a good resource for the actual equation to implement
	 *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
	 */
	for (int i = 0; i < num_particles; i++)
	{
		double paricle_x = particles[i].x;
		double paricle_y = particles[i].y;
		double paricle_theta = particles[i].theta;

		//Create a vector to hold the map landmark locations predicted to be within sensor range of the particle
		vector<LandmarkObs> predictions;

		//Each map landmark for loop
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			//Get id and x,y coordinates
			float lmx = map_landmarks.landmark_list[j].x_f;
			float lmy = map_landmarks.landmark_list[j].y_f;
			int lmid = map_landmarks.landmark_list[j].id_i;

			//Only consider landmarks within sensor range of the particle
			//(rather than using the "dist" method considering a circular region around the particle,
			//this considers a rectangular region but is computationally faster)
			if (fabs(lmx - paricle_x) <= sensor_range && fabs(lmy - paricle_y) <= sensor_range) {
				predictions.push_back(LandmarkObs{ lmid, lmx, lmy });
			}
		}

		//Create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
		vector<LandmarkObs> trans_obs;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double t_x =  paricle_x + cos(paricle_theta) * observations[j].x - sin(paricle_theta) * observations[j].y;
			double t_y = paricle_y + sin(paricle_theta) * observations[j].x + cos(paricle_theta) * observations[j].y;
			trans_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		//Data association for the predictions and transformed observations on current particle
		dataAssociation(predictions, trans_obs);
		particles[i].weight = 1.0;
		for (unsigned int j = 0; j < trans_obs.size(); j++)
		{
			double o_x, o_y, pr_x, pr_y;
			o_x = trans_obs[j].x;
			o_y = trans_obs[j].y;
			int asso_id = trans_obs[j].id;

			//x,y coordinates of the prediction associated with the current observation
			bool found_item = false;
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == asso_id) {
					found_item = true;
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			//Weight for this observation with multivariate Gaussian
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = ( 1 / (2 * M_PI * s_x * s_y)) * exp( -( pow(pr_x - o_x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - o_y, 2) / (2 * pow(s_y, 2))) ) );

			//Product of this obersvation weight with total observations weight
			particles[i].weight *= obs_w;
		}
	}

}

void ParticleFilter::resample() {
	/**
	 * TODO: Resample particles with replacement with probability proportional
	 *   to their weight.
	 * NOTE: You may find std::discrete_distribution helpful here.
	 *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	 */
	vector<double> weights;
	double maximum_Weight = numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > maximum_Weight) {
			maximum_Weight = particles[i].weight;
		}
	}

	uniform_real_distribution<double> distDouble(0.0, maximum_Weight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);
	int index = distInt(gen);
	double beta = 0.0;
	vector<Particle> resampledParticles;
	for (int i = 0; i < num_particles; i++) {
		beta = beta + distDouble(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampledParticles.push_back(particles[index]);
	}

	particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
	// particle: the particle to which assign each listed association,
	//   and association's (x,y) world coordinates mapping
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
	vector<double> v;

	if (coord == "X") {
		v = best.sense_x;
	} else {
		v = best.sense_y;
	}

	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}