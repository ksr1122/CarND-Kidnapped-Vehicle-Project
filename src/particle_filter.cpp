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

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 50;
  std::random_device rd;
  std::default_random_engine gen(rd());

  // This line creates a normal (Gaussian) distribution for x, y & theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0;i<num_particles;++i) {
    Particle particle;

    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);

    particle.weight = 1.0;
    weights.push_back(particle.weight);

    particles.push_back(particle);
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

  std::default_random_engine gen;

  for (int i=0;i<num_particles;++i) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    if (fabs(yaw_rate) < EPS) {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
    } else {
      double yaw = yaw_rate * delta_t;
      double vel_yaw = velocity / yaw_rate;
      x += vel_yaw * (sin(theta + yaw) - sin(theta));
      y += vel_yaw * (cos(theta) - cos(theta + yaw));
      theta += yaw;
    }

    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

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

  for (size_t i=0;i<observations.size();i++) {
    double min_dist = std::numeric_limits<double>::max();

    double obs_x = observations[i].x;
    double obs_y = observations[i].y;

    for (size_t j=0;j<predicted.size();j++) {
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;

      double cur_dist = dist(obs_x, obs_y, pred_x, pred_y);
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        observations[i].id = predicted[j].id;
      }
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

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  for (int i=0;i<num_particles;i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> predictions;
    for (size_t j=0;j<map_landmarks.landmark_list.size();j++) {
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;

      double cur_dist = dist(x, y, lm_x, lm_y);
      if (cur_dist <= sensor_range) {
        predictions.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, lm_x, lm_y});
      }
    }

    vector<LandmarkObs> tr_observations;
    for (size_t j=0;j<observations.size();j++) {
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

      double tr_x = x + (cos(theta) * obs_x) - (sin(theta) * obs_y);
      double tr_y = y + (sin(theta) * obs_x) + (cos(theta) * obs_y);
    
      tr_observations.push_back(LandmarkObs{observations[j].id, tr_x, tr_y});
    }

    dataAssociation(predictions, tr_observations);

    particles[i].weight = 1.0;

    for (size_t j=0;j<tr_observations.size();j++) {
      double tr_x = tr_observations[j].x;
      double tr_y = tr_observations[j].y;

      for (size_t k=0;k<predictions.size();k++) {
        if (tr_observations[j].id == predictions[k].id) {
          double pr_x = predictions[k].x;
          double pr_y = predictions[k].y;
          particles[i].weight *= multiv_prob(sig_x, sig_y, tr_x, tr_y, pr_x, pr_y);
          weights[i] = particles[i].weight;
          break;
        }
      }
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

  std::default_random_engine gen;
  std::uniform_int_distribution<int> particle_index(0, num_particles - 1);

  int index = particle_index(gen);
  double beta = 0.0;
  double mw = *max_element(weights.begin(), weights.end());

  vector<Particle> rs_particles;
  for (int i=0;i<num_particles;i++) {
    std::uniform_real_distribution<double> random_weight(0.0, 2 * mw);
    beta += random_weight(gen);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    rs_particles.push_back(particles[index]);
  }
  particles = rs_particles;
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
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
