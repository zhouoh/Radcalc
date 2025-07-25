#include <chemfiles/Atom.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include "chemfiles.hpp"
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>
#include <Eigen/Dense>
#include <array>
#include "argparse.hpp"
#include <fstream>


struct Rotor
{
    size_t pillar1;
    size_t pillar2;
    size_t rotor;
    size_t carbonyl_carbon;
    size_t carbonyl_oxygen;
    std::vector<double> dihedrals;
    std::vector<double> anglar_velocity;
    std::vector<int> AMPD_peaks;
};

Eigen::MatrixXd get_distance_matrix(chemfiles::Frame & frame, int n_atoms)
{
    Eigen::MatrixXd distance_matrix = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (size_t i = 0; i < n_atoms; i++) {
        for (size_t j = 0; j < i; j++) {
            auto distance = frame.distance(i, j);
            distance_matrix(i, j) = distance;
            distance_matrix(j, i) = distance;
        }
    }
    return distance_matrix;
}



Eigen::MatrixXd get_connectivity_matrix(std::vector<chemfiles::Bond> & bonds, int n_atoms) 
{
    Eigen::MatrixXd connectivity_matrix = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (auto bond : bonds) {
        int i = bond[0];
        int j = bond[1];
        connectivity_matrix(i, j) = 1;
        connectivity_matrix(j, i) = 1;
    }
    return connectivity_matrix;
}

// select the atoms that are bonded to the atom i
std::vector<size_t> get_bonded_atoms(int i, Eigen::MatrixXd & connectivity_matrix) 
{
    std::vector<size_t> bonded_atoms;
    for (int j = 0; j < connectivity_matrix.cols(); j++) {
        if (connectivity_matrix(i, j) == 1) {
            bonded_atoms.push_back(j);
        }
    }
    return bonded_atoms;
}

// select the carbonyl carbons
std::vector<size_t> select_carbonyl_carbons(std::vector<chemfiles::Atom> atoms, Eigen::MatrixXd & connectivity_matrix) 
{
    std::vector<size_t> carbonyl_carbons;
    for (size_t i = 0; i < atoms.size(); i++) {
        if (atoms[i].type() != "C") {
            continue;
        }
        std::vector<size_t> bonded_atoms = get_bonded_atoms(i, connectivity_matrix);
        if (bonded_atoms.size() != 3) {
            continue;
        }
        int n_oxygen = 0;
        for (auto j : bonded_atoms) {
            if (atoms[j].type() == "O") {
                n_oxygen++;
            }
        }
        if (n_oxygen == 2) {
            carbonyl_carbons.push_back(i);
        }
    }
    return carbonyl_carbons;
}

// select the quaternary carbons

std::vector<size_t> select_quaternary_carbons(std::vector<chemfiles::Atom> atoms, Eigen::MatrixXd & connectivity_matrix,std::vector<size_t> carbonyl_carbons) 
{
    std::vector<size_t> quaternary_carbons;
    for (size_t i = 0; i < atoms.size(); i++) {
        if (atoms[i].type() != "C") {
            continue;
        }
        std::vector<size_t> bonded_atoms = get_bonded_atoms(i, connectivity_matrix);
        if (bonded_atoms.size() != 4) {
            continue;
        }
        int n_carbonyl_carbon = 0;
        int n_carbon = 0;
        for (auto j : bonded_atoms) {
            if (atoms[j].type() == "C") {
                n_carbon++;
                if (std::find(carbonyl_carbons.begin(), carbonyl_carbons.end(), j) != carbonyl_carbons.end()) {
                    n_carbonyl_carbon++;
                }
            }
        }
        if (n_carbonyl_carbon == 1 && n_carbon == 4) {
            quaternary_carbons.push_back(i);
        }
    }
    return quaternary_carbons;
}

std::vector<size_t> select_carbonyl_oxygen(std::vector<chemfiles::Atom> atoms, Eigen::MatrixXd & connectivity_matrix,std::vector<size_t> carbonyl_carbons) 
{
    std::vector<size_t> carbonyl_oxygen;
    for (size_t i = 0; i < atoms.size(); i++) {
        if (atoms[i].type() != "O") {
            continue;
        }
        std::vector<size_t> bonded_atoms = get_bonded_atoms(i, connectivity_matrix);
        //if (bonded_atoms.size() != 1) {
        //    continue;
        //}
        for (auto j : bonded_atoms) {
            if (std::find(carbonyl_carbons.begin(), carbonyl_carbons.end(), j) != carbonyl_carbons.end()) {
                carbonyl_oxygen.push_back(i);
            }
        }
    }
    return carbonyl_oxygen;

}

// select rotors
std::vector<Rotor> select_rotors(std::vector<chemfiles::Atom> atoms, Eigen::MatrixXd & connectivity_matrix, std::vector<size_t> carbonyl_carbons, std::vector<size_t> quaternary_carbons, std::vector<size_t> carbonyl_oxygens, Eigen::MatrixXd & distance_matrix) 
{
    std::vector<Rotor> rotors;
    for (size_t i = 0; i < atoms.size(); i++) {
        if (atoms[i].type() != "C") {
            continue;
        }
        std::vector<size_t> bonded_atoms = get_bonded_atoms(i, connectivity_matrix);

        if (std::find(quaternary_carbons.begin(), quaternary_carbons.end(), i) != quaternary_carbons.end()) {
            continue;
        }
        if (std::find(carbonyl_carbons.begin(), carbonyl_carbons.end(), i) != carbonyl_carbons.end()) {
            continue;
        }
        Rotor rotor;
        rotor.rotor = i;
        // find the two pillars, the two pillars are the two atoms that are quaternary_carbons that are closest to the rotor
        double min_distance = 1000;
        for (size_t j = 0; j < quaternary_carbons.size(); j++) {
            if (quaternary_carbons[j] == i) {
                continue;
            }
            if (distance_matrix(i, quaternary_carbons[j]) < min_distance) {
                min_distance = distance_matrix(i, quaternary_carbons[j]);
                rotor.pillar1 = quaternary_carbons[j];
            }
        }
        // the carbonyl carbon is the atom that is bonded to the pillar1 and is not the rotor
        auto pillar1_bonded_atoms = get_bonded_atoms(rotor.pillar1, connectivity_matrix);
        for (auto j : pillar1_bonded_atoms) {
            //if j in carbonyl_carbons and j != i
            if (std::find(carbonyl_carbons.begin(), carbonyl_carbons.end(), j) != carbonyl_carbons.end() && j != i) {
                rotor.carbonyl_carbon = j;
            }
        }
        // the carbonyl oxygen is the atom that is bonded to the carbonyl carbon and is not the pillar1
        auto carbonyl_carbon_bonded_atoms = get_bonded_atoms(rotor.carbonyl_carbon, connectivity_matrix);
        for (auto j : carbonyl_carbon_bonded_atoms) {
            //if j in carbonyl_oxygens and j != pillar1
            if (std::find(carbonyl_oxygens.begin(), carbonyl_oxygens.end(), j) != carbonyl_oxygens.end() && j != rotor.pillar1) {
                rotor.carbonyl_oxygen = j;
            }
        }

        min_distance = 1000;
        for (size_t j = 0; j < quaternary_carbons.size(); j++) {
            if (quaternary_carbons[j] == i || quaternary_carbons[j] == rotor.pillar1) {
                continue;
            }
            if (distance_matrix(i, quaternary_carbons[j]) < min_distance) {
                min_distance = distance_matrix(i, quaternary_carbons[j]);
                rotor.pillar2 = quaternary_carbons[j];
            }
        }
        //std::cout << "rotor: " << rotor.rotor << " pillar1: " << rotor.pillar1 << " pillar2: " << rotor.pillar2 << std::endl;
        rotors.push_back(rotor);
    }
    return rotors;
}

double vector_cross_mutiply(chemfiles::Vector3D v1, chemfiles::Vector3D v2) 
{
    return v1[0] * v2[1] - v1[1] * v2[0];
}


void calculate_dihedral(std::vector<Rotor> & rotors, chemfiles::Frame & frame) 
{
    int rotor_num = rotors.size();
    double dihedral;
    for (int i = 0; i < rotor_num; i++)
    {
        dihedral = frame.dihedral(rotors[i].rotor, rotors[i].pillar1, rotors[i].carbonyl_carbon, rotors[i].carbonyl_oxygen);
        //convert to degree
        dihedral = dihedral * 180 / M_PI;
        rotors[i].dihedrals.push_back(dihedral);
    }
}

int get_max_length(std::vector<Rotor> & rotors)
{
    //random put 10 rotors into the data
    std::vector<double> data;
    for (int i = 0; i < 100; i++) {
        int index = rand() % rotors.size();
        for (auto d : rotors[index].dihedrals) {
            data.push_back(d);
        }
    }
    
    int count = data.size();
    std::vector<int> p_data(count, 0);
    std::vector<int> arr_rowsum;
    arr_rowsum.reserve(count / 2);

    for (int k = 1; k <= count / 2; ++k) {
        int row_sum = 0;
        for (int i = k; i < count - k; ++i) {
            if (data[i] > data[i - k] && data[i] > data[i + k]) {
                row_sum--;
            }
        }
        arr_rowsum.push_back(row_sum);
    }

    auto min_iter = std::min_element(arr_rowsum.begin(), arr_rowsum.end());
    int min_index = std::distance(arr_rowsum.begin(), min_iter);
    int max_window_length = min_index;
    return max_window_length;
}

void AMPD(Rotor & rotor, int max_window_length=1000)
{
    std::vector<double> data = rotor.dihedrals;
    int count = data.size();
    //std::cout << "count: " << count << std::endl;
    std::vector<int> p_data(count, 0);
    std::vector<int> arr_rowsum;


    for (int k = 1; k <= max_window_length; ++k) {
        for (int i = k; i < count - k; ++i) {
            if (data[i] > data[i - k] && data[i] > data[i + k]) {
                p_data[i]++;
            }
        }
    }
    //std::cout<<"1"<<std::endl;

    std::vector<int> peaks;
    for (int i = 0; i < count; ++i) {
        if (p_data[i] == max_window_length) {
            if (data[i] < 170.0){
                continue;
            }
            peaks.push_back(i);
            //std::cout << "peak: " << i << " " << data[i] << std::endl;
        }
    }
    if (peaks.size() < 2) {
        rotor.AMPD_peaks = peaks;
        return;
    }
    //std::cout<<"2"<<std::endl;
    // check the average of the data between two peaks
    for (int i = 0; i < peaks.size() - 1; i++) {
        int start = peaks[i];
        int end = peaks[i + 1];
        double sum = 0.0;
        for (int j = start; j < end; j++) {
            sum += data[j];
        }
        double average = sum / (end - start);
        //std::cout << "average: " << average << std::endl;
        if (average > 100.0) {
            peaks.erase(peaks.begin() + i);
        }
    }
    //std::cout<<"3"<<std::endl;

    rotor.AMPD_peaks = peaks;
}

void calculate_anglar_velocity_sob(std::vector<Rotor> & rotors, double dt, int interval = 1) 
{
    int rotor_num = rotors.size();
    double anglar_velocity;
    for (int i = 0; i < rotor_num; i++)
    {
        int frame_num = rotors[i].dihedrals.size();
        //loop over all frames
        for (int j = interval; j < frame_num; j++)
        {
            anglar_velocity = (rotors[i].dihedrals[j] - rotors[i].dihedrals[j-interval]) / (dt * interval);
            rotors[i].anglar_velocity.push_back(anglar_velocity);
        }
    }
}

std::vector<double> get_sob_anglar_velocity(std::vector<Rotor> & rotors) 
{
    std::vector<double> anglar_velocity;
    int rotor_num = rotors.size();
    for (int i = 0; i < rotor_num; i++)
    {
        int frame_num = rotors[i].anglar_velocity.size();
        for (int j = 0; j < frame_num; j++)
        {
            anglar_velocity.push_back(rotors[i].anglar_velocity[j]);
        }
    }
    return anglar_velocity;
}

std::array<double,2> calculate_anglar_velocity(Rotor rotor, chemfiles::Frame & frame) 
{
    auto positions = frame.positions();
    auto velocities = frame.velocities();
    chemfiles::span<chemfiles::Vector3D> velocities_span = velocities.value();
    double vx = velocities_span[rotor.rotor][0];
    double vy = velocities_span[rotor.rotor][1];
    double vz = velocities_span[rotor.rotor][2];
    double v = sqrt(vx * vx + vy * vy + vz * vz);
    chemfiles::Vector3D v_vector = {vx, vy, vz};
    auto pillar1 = positions[rotor.pillar1];
    auto pillar2 = positions[rotor.pillar2];
    auto rotor_position = positions[rotor.rotor];
    chemfiles::Vector3D pillar1_rotor = pillar1 - rotor_position;
    chemfiles::Vector3D pillar2_rotor = pillar2 - rotor_position;
    chemfiles::Vector3D pillar1_pillar2 = pillar2 - pillar1;
    double pillar1_rotor_norm = sqrt(pillar1_rotor[0] * pillar1_rotor[0] + pillar1_rotor[1] * pillar1_rotor[1] + pillar1_rotor[2] * pillar1_rotor[2]);
    chemfiles::Vector3D product = chemfiles::cross(pillar1_rotor, pillar2_rotor);
    double product_norm = sqrt(product[0] * product[0] + product[1] * product[1] + product[2] * product[2]);
    double linear_velocity = chemfiles::dot(v_vector, product) / product_norm;
    linear_velocity  =  abs(linear_velocity);
    //std::cout << "linear velocity: " << linear_velocity << std::endl;
    // 
    double projection = chemfiles::dot(pillar1_pillar2, pillar1_rotor) / sqrt(pillar1_pillar2[0] * pillar1_pillar2[0] + pillar1_pillar2[1] * pillar1_pillar2[1] + pillar1_pillar2[2] * pillar1_pillar2[2]);
    double cos_theta = abs(projection) / pillar1_rotor_norm;
    double theta = acos(cos_theta);
    double radius = pillar1_rotor_norm * sin(theta);

    double omega = abs(linear_velocity) / radius;
    

    return {omega, linear_velocity};
}


std::vector<std::vector<double>> calculate_distribution(std::vector<double> & v, int n_bins, double imin = 0.0, double imax = 0.0)
{
    std::vector<std::vector<double>> distribution;
    double max,min;
    //check if imin and imax are given
    if (imin == 0.0 && imax == 0.0) {
        max = *std::max_element(v.begin(), v.end());
        
        max = max*1.1;

        min = 0.0;
    }
    else {
        max = imax;
        min = imin;
    }



    double interval = (max - min) / n_bins;

    //intialize the distribution
    for (int i = 0; i < n_bins; i++) {
        distribution.push_back({min + interval * i, 0.0});
    }
    //std::cout << "distribution size: " << distribution.size() << std::endl;
    for (int i = 0; i < v.size(); i++) {
        int index = int((v[i] - min) / interval);
        if (index == n_bins) {
            index = n_bins - 1;
        }
        if (index < 0) {
            continue;   
        }
        if (index >= distribution.size()) {
            //std::cout << "index: " << index << std::endl;
            continue;
        }
        //std::cout << "index: " << index << std::endl;
        distribution[index][1] += 1;
    }
    return distribution;
}


std::vector<std::vector<double>> calculate_distribution(std::vector<int> & v, int n_bins, double imin = 0.0, double imax = 0.0)
{
    std::vector<std::vector<double>> distribution;
    double max,min;
    //check if imin and imax are given
    if (imin == 0.0 && imax == 0.0) {
        max = *std::max_element(v.begin(), v.end());
        
        max = max*1.1;

        min = 0.0;
    }
    else {
        max = imax;
        min = imin;
    }



    double interval = (max - min) / n_bins;

    //intialize the distribution
    for (int i = 0; i < n_bins; i++) {
        distribution.push_back({min + interval * i, 0.0});
    }
    //std::cout << "distribution size: " << distribution.size() << std::endl;
    for (int i = 0; i < v.size(); i++) {
        int index = int((v[i] - min) / interval);
        if (index == n_bins) {
            index = n_bins - 1;
        }
        if (index < 0) {
            continue;   
        }
        if (index >= distribution.size()) {
            //std::cout << "index: " << index << std::endl;
            continue;
        }
        //std::cout << "index: " << index << std::endl;
        distribution[index][1] += 1;
    }
    return distribution;
}

double average(std::vector<double> & v) 
{
    double sum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum / v.size();
}

double median(std::vector<double> & v) 
{
    std::sort(v.begin(), v.end());
    int n = v.size();
    if (n % 2 == 0) {
        return (v[n/2] + v[n/2 - 1]) / 2;
    } else {
        return v[n/2];
    }
}

std::vector<double> read_dat(std::string filename)
{
    //read the second column of data from a file
    std::vector<double> data;
    std::ifstream fin;
    fin.open(filename);
    double f,d;
    while (fin >> f >> d) {
        data.push_back(d);
    }
    fin.close();

    return data;
}

double traj_rms(Rotor rotor, std::vector<double> reference)
{
    std::vector<double> traj = rotor.dihedrals;
    double sum = 0.0;
    // get the smaller size
    int size = traj.size() < reference.size() ? traj.size() : reference.size();
    for (int i = 0; i < size; i++) {
        sum += (std::abs(traj[i]) - std::abs(reference[i])) * (std::abs(traj[i]) - std::abs(reference[i]));
    }
    return sqrt(sum / size);
}


int main(int argc, char const *argv[]) {

    auto args = util::argparser("Calculate the radial velocity distribution of rotors");
    args.set_program_name("RadVcalc")
    .add_help_option()
    .add_sc_option("-v", "--version", "show version info", []() {
            std::cout << "version " << "0.1.0" << std::endl;
        })
    .add_option<std::string>("-i", "--input", ".trr trajectory file", "traj_v.trr")
    .add_option<std::string>("-p", "--top", "tpr file", "traj_v.tpr")
    .add_option<std::string>("-f", "--datfile", ".dat file", "multiplot.dat")
    .add_option<int>("-c", "--compare", "compare traj", 0)
    .add_option<int>("-n", "--nbins", "number of bins", 100)
    .add_option<double>("-d", "--dt", "time interval", 0.1)
    .add_option<double>("-m", "--min", "minimum value", 0.0)
    .add_option<double>("-M", "--max", "maximum value", 0.0)
    .add_option<int>("-s", "--step", "step interval", 1)
    .parse(argc, argv);

    

    std::string input = args.get_option<std::string>("--input");
    std::string prefix = input.substr(0, input.find_last_of(".")); 
    std::string top = args.get_option<std::string>("--top");
    int compare = args.get_option<int>("--compare");
    int n_bins = args.get_option<int>("--nbins");

    chemfiles::Trajectory trajectory(input);
    std::cout << "There are " << trajectory.nsteps() << " steps in the trajectory" << std::endl;
    trajectory.set_topology(top);
    auto frame = trajectory.read();
    std::cout << "There are " << frame.size() << " atoms in the frame" << std::endl;

    chemfiles::Topology topology = frame.topology();
    std::cout << "The topology has " << topology.size() << " atoms" << std::endl;

    // Save a pdb file for the first frame
    /*
    auto first_frame = trajectory.read();
    chemfiles::Trajectory output("first_frame.pdb", 'w');
    output.write(first_frame);
    output.close();
    std::cout << "The first frame is saved to first_frame.pdb" << std::endl;
    */
    std::vector<chemfiles::Atom> atoms;
    for (auto atom : topology) {
        atoms.push_back(atom);
    }
    // Do awesome science with the positions here !
    std::vector<chemfiles::Bond> bonds = topology.bonds();
    
    Eigen::MatrixXd connectivity_matrix = get_connectivity_matrix(bonds, topology.size());
    Eigen::MatrixXd distance_matrix = get_distance_matrix(frame, topology.size());
    std::vector<size_t> carbonyl_carbons = select_carbonyl_carbons(atoms, connectivity_matrix);
    std::cout << "There are " << carbonyl_carbons.size() << " carbonyl carbons" << std::endl;
    std::vector<size_t> quaternary_carbons = select_quaternary_carbons(atoms, connectivity_matrix, carbonyl_carbons);
    std::cout << "There are " << quaternary_carbons.size() << " quaternary carbons" << std::endl;
    std::vector<size_t> carbonyl_oxygens = select_carbonyl_oxygen(atoms, connectivity_matrix, carbonyl_carbons);
    std::cout << "There are " << carbonyl_oxygens.size() << " carbonyl oxygens" << std::endl;
    std::vector<Rotor> rotors = select_rotors(atoms, connectivity_matrix, carbonyl_carbons, quaternary_carbons, carbonyl_oxygens, distance_matrix);
    std::cout << "There are " << rotors.size() << " rotors" << std::endl;

    //output the rotors to file
    std::ofstream fout;
    fout.open("rotors.txt");
    for (auto rotor : rotors) {
        fout << rotor.pillar1 << " " << rotor.pillar2 << " " << rotor.rotor << " " << rotor.carbonyl_carbon << " " << rotor.carbonyl_oxygen << std::endl;
    }
    fout.close();

    std::vector<double> omega;
    std::vector<double> linear_velocity;

    for (int i =0; i < trajectory.nsteps(); i++) {
        auto frame = trajectory.read_step(i);
        calculate_dihedral(rotors, frame);
    }

    if (compare == 0){

        // calculate omega and linear velocity for all frames
        for (int i =0; i < trajectory.nsteps(); i++) {
            auto frame = trajectory.read_step(i);
            // console progress bar
            std::cout << "\r" << "Progress: " << i << "/" << trajectory.nsteps() << std::flush;
            for (auto rotor : rotors) {
                //std::cout << "rotor: " << rotor.pillar1 << " " << rotor.pillar2 << " " << std::endl;
                auto result = calculate_anglar_velocity(rotor, frame);
                
                omega.push_back(result[0]);
                linear_velocity.push_back(result[1]);
                
            }
        }

        std::cout << std::endl;
        std::cout << "Calculating anglar velocity by Sob ..." << std::endl;
        double dt = args.get_option<double>("--dt");
        int interval = args.get_option<int>("--step");
        calculate_anglar_velocity_sob(rotors, dt, interval);
        std::vector<double> sob_anglar_velocity = get_sob_anglar_velocity(rotors);
        std::vector<double> abs_sob_anglar_velocity;
        for (auto v : sob_anglar_velocity) {
            abs_sob_anglar_velocity.push_back(abs(v));
        }

        std::cout << std::endl;
        std::cout << "In total, there are " << omega.size() << " values" << std::endl;
        std::cout << "Writing to file..." << std::endl;

        fout.open(prefix + "_omega.txt");
        int i = 0;
        for (auto w : omega) {
            fout << w << " ";
            i++;
            if (i % 50 == 0) {
                fout << std::endl;
            }
        }
        fout.close();
        std::cout << "Writing to file..." << std::endl;

        fout.open(prefix + "_linear_velocity.txt");
        i = 0;
        for (auto v : linear_velocity) {
            fout << v << " ";
            i++;
            if (i % 50 == 0) {
                fout << std::endl;
            }
        }
        fout.close();

        fout.open(prefix + "_sob_anglar_velocity.txt");
        i = 0;
        for (auto v : sob_anglar_velocity) {
            fout << v << " ";
            i++;
            if (i % 50 == 0) {
                fout << std::endl;
            }
        }
        fout.close();

        fout.open(prefix + "_abs_sob_anglar_velocity.txt");
        i = 0;
        for (auto v : abs_sob_anglar_velocity) {
            fout << v << " ";
            i++;
            if (i % 50 == 0) {
                fout << std::endl;
            }
        }
        fout.close();

        std::cout << "Calculating distribution..." << std::endl;

        double min = args.get_option<double>("--min");
        double max = args.get_option<double>("--max");

        auto omega_distribution = calculate_distribution(omega, n_bins, 0.0, max);
        auto linear_velocity_distribution = calculate_distribution(linear_velocity, n_bins, 0.0, max);
        auto sob_anglar_velocity_distribution = calculate_distribution(sob_anglar_velocity, n_bins, min, max);
        auto abs_sob_anglar_velocity_distribution = calculate_distribution(abs_sob_anglar_velocity, n_bins, 0.0, max);
        std::cout << "Writing to file..." << std::endl;
        fout.open(prefix + "_omega_distribution.txt");
    
        for (auto w : omega_distribution) {
            fout << w[0] << "  " << w[1] << std::endl;
        }
        fout.close();

        fout.open(prefix + "_linear_velocity_distribution.txt");
        for (auto v : linear_velocity_distribution) {
            fout << v[0] << "  " << v[1] << std::endl;
        }
        fout.close();

        fout.open(prefix + "_sob_anglar_velocity_distribution.txt");
        for (auto v : sob_anglar_velocity_distribution) {
            fout << v[0] << "  " << v[1] << std::endl;
        }
        fout.close();

        fout.open(prefix + "_abs_sob_anglar_velocity_distribution.txt");
        for (auto v : abs_sob_anglar_velocity_distribution) {
            fout << v[0] << "  " << v[1] << std::endl;
        }
        fout.close();

        std::cout << "Calculating average and median..." << std::endl;
        std::cout << "Average omega: " << average(omega) << std::endl;
        std::cout << "Median omega: " << median(omega) << std::endl;
        std::cout << "Average linear velocity: " << average(linear_velocity) << std::endl;
        std::cout << "Median linear velocity: " << median(linear_velocity) << std::endl;
        std::cout << std::endl;
        std::cout << "Average sob anglar velocity: " << average(sob_anglar_velocity) << std::endl;
        std::cout << "Median sob anglar velocity: " << median(sob_anglar_velocity) << std::endl;
        std::cout << "Average abs sob anglar velocity: " << average(abs_sob_anglar_velocity) << std::endl;
        std::cout << "Median abs sob anglar velocity: " << median(abs_sob_anglar_velocity) << std::endl;
    }
    else if (compare == 1) {
        std::cout << "Comparing traj..." << std::endl;
        std::string compare_traj = args.get_option<std::string>("--datfile");
        std::vector<double> reference = read_dat(compare_traj);
        std::cout << "There are " << reference.size() << " values in the reference traj" << std::endl;
        std::vector<double> rms;
        for (auto rotor : rotors) {
            rms.push_back(traj_rms(rotor, reference));
        }
        std::cout << "Writing to file..." << std::endl;
        fout.open(prefix + "_rms.txt");
        for (auto r : rms) {
            fout << r << std::endl;
        }
        fout.close();
        //get the minium rms and the corresponding rotor
        double min_rms = *std::min_element(rms.begin(), rms.end());
        int min_index = std::distance(rms.begin(), std::min_element(rms.begin(), rms.end()));
        std::cout << "The minium rms is " << min_rms << " and the corresponding rotor is " << min_index << std::endl;
        //output the rotor to screen
        std::cout << "The rotor is " << rotors[min_index].pillar1 << " " << rotors[min_index].pillar2 << " " << rotors[min_index].rotor << " " << rotors[min_index].carbonyl_carbon << " " << rotors[min_index].carbonyl_oxygen << std::endl;
    
    }
    else if (compare == 2){
        int index = n_bins;
        //output the dihedrals of the first rotor to file
        std::cout << "Writing to file..." << std::endl;
        fout.open(prefix + "_dihedrals.txt");
        for (auto d : rotors[index].dihedrals) {
            fout << d << "\n";
        }
        fout.close();
        //int max_window_length = get_max_length(rotors);
        //std::cout << "max window length: " << max_window_length << std::endl;
        AMPD(rotors[index], 40);
        fout.open(prefix + "_AMPD_peaks.txt");
        for (auto p : rotors[index].AMPD_peaks) {
            fout << p << "\n";
        }
        fout.close();


    }
    else if (compare == 3){
        int index = 0;
        for (auto& rotor : rotors) {
            AMPD(rotor,n_bins);
            //show the progress
            std::cout << "\r" << "Progress: " << index << "/" << rotors.size() << std::flush;
            index++;
        }
        std::cout << "Here are the AMPD peaks for each rotor" << std::endl;
        std::vector<int> intervals;
        for (auto rotor : rotors) {
            //int size = rotor.AMPD_peaks.size();
            //std::cout << size << " ";
            //std::cout << "rotor: " << rotor.pillar1 << " " << rotor.pillar2 << " " << rotor.rotor << " " << rotor.carbonyl_carbon << " " << rotor.carbonyl_oxygen << std::endl;
            if (rotor.AMPD_peaks.size() < 2) {
                continue;
            }

            for (int i = 0; i < rotor.AMPD_peaks.size() - 1; i++) {
                
                intervals.push_back(rotor.AMPD_peaks[i + 1] - rotor.AMPD_peaks[i]);
            }
        }
        //get histogram of intervals
        std::vector<std::vector<double>> interval_distribution = calculate_distribution(intervals, 100, 0.0, 12.0/0.05);
        //output the intervals to file
        std::cout << "Writing to file..." << std::endl;
        fout.open(prefix + "_intervals.txt");
        for (auto i : intervals) {
            fout << i << "\n";
        }
        fout.close();
        fout.open(prefix + "_interval_distribution.txt");
        for (auto i : interval_distribution) {
            fout << i[0] << "  " << i[1] << std::endl;
        }
    }
    else if (compare == 4){
        int index = 0;
        for (auto& rotor : rotors) {
            AMPD(rotor,n_bins);
            //show the progress
            std::cout << "\r" << "Progress: " << index << "/" << rotors.size() << std::flush;
            index++;
        }
        std::cout << "Here are the AMPD peaks for each rotor" << std::endl;
        std::vector<double> freq;
        for (auto rotor : rotors) {
            //int size = rotor.AMPD_peaks.size();
            //std::cout << size << " ";
            //std::cout << "rotor: " << rotor.pillar1 << " " << rotor.pillar2 << " " << rotor.rotor << " " << rotor.carbonyl_carbon << " " << rotor.carbonyl_oxygen << std::endl;
            if (rotor.AMPD_peaks.size() < 2) {
                freq.push_back(0);
                continue;
            }
            double f = static_cast<double>(rotor.dihedrals.size()) / (rotor.AMPD_peaks.size() - 1);
            freq.push_back(f);
        }
        //output the freq to file
        std::cout << "Writing to file..." << std::endl;
        fout.open(prefix + "_freq.txt");
        for (auto f : freq) {
            fout << f << "\n";
        }
        fout.close();
    }

    return 0;
}


