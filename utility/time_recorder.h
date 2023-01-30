/**
 * @file time_recorder.h
 * @brief contains an logic class TimeRecorder
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.28
 */

#ifndef TIME_RECORDER_H
#define TIME_RECORDER_H

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono> // NOLINT [build/c++11]
#include "defines.h"

using std::vector;
using std::string;
using std::find;

/*!
* An logic class that records time
*/
class TimeRecorder {
public:
	vector<string> events_;
	vector<double> times_;
	vector<TimePoint> start_;
	vector<TimePoint> end_;

	void event_start(string event_name) {
		events_.push_back(event_name);
		TimePoint start = std::chrono::high_resolution_clock::now();
		start_.push_back(start);
		times_.resize(events_.size(), 0);
		end_.resize(events_.size());
	}
	void event_end(string event_name) {
		TimePoint end = std::chrono::high_resolution_clock::now();
		auto it = find(events_.begin(), events_.end(), event_name);
		if (it != events_.end()) {
			int index = it - events_.begin();
        	end_[index] = end;
        	times_[index] = std::chrono::duration_cast<std::chrono::duration<double>>(end - start_[index]).count() * 1000;
		} else {
			cout << "warning: no event " << event_name << endl;
		}
	}

	void print_all_events() {
		for (int i = 0; i < events_.size(); i++) {
			cout << "time cost(" << events_[i] << "): " << times_[i] << "ms" << endl;
		}
	}
};

#endif