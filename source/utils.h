#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

void format1Cout(vector<double>& v, string vector_name) {
    cout << "---------------------------------------------------------" << endl;
    cout << "Elements of " << vector_name << endl;
    for (int i = 0; i < 4; i++) {
        cout << v[i] << " ";
    }
    cout << "... ";
    for (int i = v.size() - 5; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;
    cout << endl;
    cout << "Length of " << vector_name << " - " << v.size() << endl;
    cout << "---------------------------------------------------------" << endl;
}

void format2Cout(vector<vector<double>>& v, string table_name) {
    int len = v[0].size();
    cout << "---------------------------------------------------------" << endl;
    cout << "Elements of " << table_name << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << v[i][j] << " ";
        }
        cout << "... ";
        for (int j = len - 5; j < len; j++) {
            cout << v[i][j] << " ";
        }
        cout << endl;
    }
    cout << "......................................." << endl;
    for (int i = v.size() - 5; i < v.size(); i++) {
        for (int j = 0; j < 4; j++) {
            cout << v[i][j] << " ";
        }
        cout << "... ";
        for (int j = len - 5; j < len; j++) {
            cout << v[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "Shape of " << table_name <<  " - " << "(" << v.size() << ", " << v[0].size() << ")" << endl;
    cout << "---------------------------------------------------------" << endl;
}

double oneDimProd(const vector<double>& v) {
    double result = 1.0;
    for (size_t i = 0; i < v.size(); ++i) {
        result *= v.at(i);
    }
    return result;
}

vector<double> oneDimScalarMultiplication(const vector<double>& v, double scalar) {
    vector<double> result;
    for (size_t i = 0; i < v.size(); ++i) {
        result.push_back(v.at(i) * scalar);
    }
    return result;
}

vector<vector<double>> multiplication(const vector<vector<double>>& v1, const vector<vector<double>>& v2) {
    vector<vector<double>> result(v1.size(), vector<double>(v2.at(0).size()));
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v2.at(0).size(); ++j) {
            for (size_t k = 0; k < v1.at(0).size(); ++k) {
                result.at(i).at(j) += v1.at(i).at(k) * v2.at(k).at(j);
            }
        }
    }
    return result;
}

vector<double> oneDimZeros(int size) {
    vector<double> zeros;
    for (size_t i = 0; i < size; ++i) {
        zeros.push_back(0.0);
    }
    return zeros;
}

vector<vector<double>> twoDimZeros(int ax1, int ax2) {
    vector<vector<double>> zeros(ax1, vector<double>(ax2));
    for (size_t i = 0; i < ax1; ++i) {
        for (size_t j = 0; j < ax2; ++j) {
            zeros.at(i).at(j) = 0.0;
        }
    }
    return zeros;
}

vector<double> append(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        result.push_back(v1.at(i));
    }
    for (size_t i = 0; i < v2.size(); ++i) {
        result.push_back(v2.at(i));
    }
    return result;
}

vector<double> arange(int size, int step) {
    vector<double> result;
    int i = 0;
    while (i < size) {
        result.push_back(i);
        i += step;
    }
    return result;
}

vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    vector<vector<double>> result(matrix.at(0).size(), vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.at(0).size(); ++i) {
        for (size_t j = 0; j < matrix.size(); ++j) {
            result.at(i).at(j) = matrix.at(j).at(i);
        }
    }
    return result;
}

vector<vector<double>> tile(const vector<double>& v, int ax1, int ax2) {
    vector<vector<double>> result(ax1, vector<double>(ax2 * v.size()));
    for (size_t i = 0; i < ax1; ++i) {
        for (size_t j = 0; j < ax2; ++j) {
            for (size_t k = 0; k < v.size(); ++k) {
                result.at(i).at(j * v.size() + k) = v.at(k);
            }
        }
    }
    return result;
}

vector<double> oneDimAddition(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        result.push_back(v1.at(i) + v2.at(i));
    }
    return result;
}

vector<vector<double>> twoDimAddition(const vector<vector<double>>& v1, const vector<vector<double>>& v2) {
    vector<vector<double>> result(v1.size(), vector<double>(v1.at(0).size()));
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v1.at(0).size(); ++j) {
            result.at(i).at(j) = v1.at(i).at(j) + v2.at(i).at(j);
        }
    }
    return result;
}

vector<vector<double>> twoDimOneDimdot(const vector<vector<double>>& v1, const vector<double>& v2) {
    vector<vector<double>> result(v1.size(), vector<double>(v1.at(0).size()));
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v1.at(0).size(); ++j) {
            result.at(i).at(j) = v1.at(i).at(j) * v2.at(j);
        }
    }
    return result;
}

vector<vector<double>> twoDimDot(const vector<vector<double>>& v1, const vector<vector<double>>& v2) {
    vector<vector<double>> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v1.at(0).size(); ++j) {
            result.at(i).at(j) = v1.at(i).at(j) * v2.at(i).at(j);
        }
    }
    return result;
}

vector<double> rfft(const vector<double>& signal, int size) {
    vector<double> temp;
    for (size_t i = 0; i < signal.size(); ++i) {
        temp.push_back(signal.at(i));
    }
    if (signal.size() < size) {
        for (size_t i = 0; i < size - signal.size(); ++i) {
            temp.push_back(0);
        }
    }
    if (signal.size() > size) {
        for (size_t i = size; i < signal.size(); ++i) {
            temp.at(i) = 0;
        }
    } 
    int new_size = (size / 2) + 1;
    vector<double> rfft_signal;
    for (size_t i = 0; i < new_size; ++i) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (size_t j = 0; j < size; ++j) {
            sum_real += temp.at(j)*cos((2*M_PI*i*j)/size);
            sum_imag += temp.at(j)*sin((2*M_PI*i*j)/size);
        }
        double modulus = sqrt(pow(sum_real, 2) + pow(sum_imag, 2));
        rfft_signal.push_back(modulus);
    }
    return rfft_signal;
}

vector<double> linspace(double start, double end, int num) {
    double step = (end - start) / (num - 1);
    vector<double> lins;
    for (size_t i = 0; i < num; ++i) {
        lins.push_back(start + (double)i * step);
    }
    return lins;
}

vector<double> cosArray(const vector<double>& v, double alpha) {
    vector<double> result;
    for (size_t i = 0; i < v.size(); ++i) {
        result.push_back(cos(alpha * v.at(i)));
    }
    return result;
}

vector<double> constantOneDimPad(const vector<double>& v, int n) {
    vector<double> result;
    int pad = v.size() - n;
    int pad1;
    int pad2;
    if (pad % 2 == 0) {
        pad1 = pad / 2;
        pad2 = pad / 2;
    }
    else {
        pad1 = (int)(pad / 2);
        pad2 = pad2 + 1;
    }
    for (size_t i = 0; i < pad1; ++i) {
        result.push_back(0);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        result.push_back(v.at(i));
    }
    for (size_t i = 0; i < pad2; ++i) {
        result.push_back(0);
    }
    return result;
}

vector<double> reverse(const vector<double>& v) {
    vector<double> result;
    for (size_t i = v.size(); i > 0; --i) {
        result.push_back(v.at(i - 1));
    }
    return result;
}

vector<vector<double>> expandDim(const vector<double>& v, int ax1, int ax2) {
    if (ax1 == -1) {
        vector<vector<double>> result(v.size(), vector<double>(1));
        for (size_t i = 0; i < v.size(); ++i) {
            result.at(i).at(0) = v.at(i);
        }
        return result;
    }
    else {
        vector<vector<double>> result(1, vector<double>(v.size()));
        for (size_t i = 0; i < v.size(); ++i) {
            result.at(0).at(i) = v.at(i);
        }
        return result;
    }  
}

vector<double> diff(const vector<double>& v) {
    vector<double> diff_v;
    for (size_t i = 1; i < v.size(); ++i) {
        diff_v.push_back(v.at(i) - v.at(i - 1));
    }
    return diff_v;
}

double oneDimMaximum(const vector<double>& v) {
    double max = v.at(0);
    for (size_t i = 1; i < v.size(); ++i) {
        if (v.at(i) > max) {
            max = v.at(i);
        }
    }
    return max;
}

double oneDimMinimum(const vector<double>& v) {
    double min = v.at(0);
    for (size_t i = 1; i < v.size(); ++i) {
        if (v.at(i) < min) {
            min = v.at(i);
        }
    }
    return min;
}

vector<vector<double>> subtractOuter(const vector<double>& v1, const vector<double>& v2) {
    vector<vector<double>> outers(v1.size(), vector<double>(v2.size()));
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v2.size(); ++j) {
            outers.at(i).at(j) = v1.at(i) - v2.at(j);
        }
    }
    return outers;
}

vector<double> compareOneDimMinimum(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1.at(i) < v2.at(i)) {
            result.push_back(v1.at(i));
        }
        else {
            result.push_back(v2.at(i));
        }
    }
    return result;
}

vector<double> compareOneDimMaximum(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1.at(i) > v2.at(i)) {
            result.push_back(v1.at(i));
        }
        else {
            result.push_back(v2.at(i));
        }
    }
    return result;
}

vector<double> oneDimDot(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result;
    for (size_t i = 0; i < v1.size(); ++i) {
        result.push_back(v1.at(i) * v2.at(i));
    }
    return result;
}

vector<double> reflectOneDimPad(const vector<double>& v, int n) {
    vector<double> result;

    vector<double> forward;
    vector<double> backward;

    for (int i = 1; i < v.size(); i++) {
        forward.push_back(v[i]);
    }
    for (int i = v.size() - 2; i >= 0; i--) {
        forward.push_back(v[i]);
    }

    int fool = (int)(n / forward.size());
    int tail = n % forward.size();

    vector<double> final_forward;
    for (int i = 0; i < fool; i++) {
        for (int j = 0; j < forward.size(); j++) {
            final_forward.push_back(forward[j]);
        }
    }
    for (int i = 0; i < tail; i++) {
        final_forward.push_back(forward[i]);
    }
    final_forward = reverse(final_forward);

    for (int i = v.size() - 2; i >= 0; i--) {
        backward.push_back(v[i]);
    }
    for (int i = 1; i < v.size(); i++) {
        backward.push_back(v[i]);
    }

    vector<double> final_backward;
    for (int i = 0; i < fool; i++) {
        for (int j = 0; j < backward.size(); j++) {
            final_backward.push_back(backward[j]);
        }
    }
    for (int i = 0; i < tail; i++) {
        final_backward.push_back(backward[i]);
    }

    result = append(final_forward, v);
    result = append(result, final_backward);

    return result;
}

