"""
This module contains classes to handle datasets consisting of many files. They
are inspired by the implemented dataset classes in atmlab written by Gerrit
Holl.

Created by John Mrziglod, June 2017
"""

import atexit
from collections import defaultdict, Iterable
from datetime import datetime, timedelta
import glob
from itertools import tee
import json
from multiprocessing import Pool
import numbers
import os.path
import re
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import typhon.files
import typhon.plots
from typhon.spareice.array import ArrayGroup
from typhon.spareice.handlers import CSV, FileInfo, NetCDF4
from typhon.trees import IntervalTree

__all__ = [
    "Dataset",
    "DatasetManager",
]


class NoFilesError(Exception):
    """Should be raised if no files were found by the :meth:`find_files`
    method.

    """
    def __init__(self, name, start, end, *args):
        message = \
            "Found no files for %s between %s and %s!\nMaybe you "\
            "misspelled the files path? Or maybe there are "\
            "no files for this time period?" % (name, start, end)
        Exception.__init__(self, message, *args)


class NoHandlerError(Exception):
    """Should be raised if no file handler is specified in a dataset object but
    a handler is required.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


class InhomogeneousFilesError(Exception):
    """Should be raised if the files of a dataset do not have the same internal
    structure but it is required.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


class Dataset:
    """Class which provides methods to handle a set of multiple files
    (dataset).

    """
    placeholder = {
        # "placeholder_name": [regex to find the placeholder]
        "year": "(\d{4})",
        "year2": "(\d{2})",
        "month": "(\d{2})",
        "day": "(\d{2})",
        "doy": "(\d{3})",
        "hour": "(\d{2})",
        "minute": "(\d{2})",
        "second": "(\d{2})",
        "millisecond": "(\d{3})",
        "end_year": "(\d{4})",
        "end_year2": "(\d{2})",
        "end_month": "(\d{2})",
        "end_day": "(\d{2})",
        "end_doy": "(\d{3})",
        "end_hour": "(\d{2})",
        "end_minute": "(\d{2})",
        "end_second": "(\d{2})",
        "end_millisecond": "(\d{3})",
        "name": "",
    }

    _temporal_resolution = {
        # placeholder: [pandas frequency, rank]
        "year": ["1A", 0,],
        "year2": ["1A", 0, ],
        "month": ["1M", 1,],
        "day": ["1D", 2,],
        "doy": ["1D", 2,],
        "hour": ["1H", 3],
        "minute": ["1T", 4],
        "second": ["1S", 5],
    }

    def __init__(
            self, files, handler=None, name=None, time_coverage=None,
            times_cache=None, continuous=True, max_processes=None,
            compress=True, decompress=True,
    ):
        """Initializes a dataset object.

        Args:
            files: A string with the complete path to the dataset files. The
                string can contain placeholder such as {year}, {month},
                etc. See the documentation for the :meth:`generate_filename`
                method for a complete list. If no placeholders are given, the
                path must point to a file. This dataset is then seen as a
                single file dataset.
            name: The name of the dataset.
            handler: An object which can handle the dataset files.
                This dataset class does not care which format its files have
                when this file handler object is given. You can use a file
                handler class from typhon.handlers or write your own class.
                For example, if this dataset consists of NetCDF files, you can
                use the typhon.spareice.handlers.NetCDF4 here (is default).
            time_coverage: Defines how the timestamp of the data
                should be retrieved. Default for multi file datasets is
                *filename* which retrieves the time coverage from the filename
                by using placeholders.
                Another option is *content* which uses the :meth:`get_info`
                method (a file handler has to be specified). The third option
                should be used if this dataset consists of a single file: then
                you can specify a tuple of two datetime objects via this
                parameter representing the start and end time. Otherwise the
                year 1 and 9999 will be used a default time coverage.
                Look at Dataset.retrieve_timestamp() for more details.
            times_cache: Finding the correct files for a time period
                may take a while, especially when the time retrieving
                method is set to "content". Therefore, if the file names and
                their time coverage are cached, multiple calls of find_files
                (for time coverages that are close) are significantly faster.
                Specify a name to a file here (which need not exist) if you
                wish to save those time coverages to a file. When restarting
                your script, this cache is used.
            continuous: If true, all files of this dataset are considered to be
                continuous, i.e. they cover a time period and not only a single
                timestamp. If their start and end time are equal,
                the minimal time resolution will be added to the end time. The
                minimal time resolution is retrieved from the temporal
                placeholders in the *files* parameter. This will be ignored if
                *time_coverage* is not *filename*.
            max_processes: Maximal number of parallel processes that will be
                used for :meth:`~typhon.spareice.datasets.Dataset.map` or
                :meth:`~typhon.spareice.datasets.Dataset.map_content` like
                methods per default (default is the number of CPUs).
            compress: If true and the *files* path ends with a compression
                suffix (such as *.zip*, *.gz*, *.b2z*, etc.), newly created
                dataset files will be compressed after writing them to disk.
                Default value is true.
            decompress: If true and the *files* path ends with a compression
                suffix (such as *.zip*, *.gz*, *.b2z*, etc.), dataset files
                will be decompressed before reading them. Default value is
                true.

        Examples:

        .. code-block:: python

            ### Multi file dataset ###
            # Define a dataset consisting of multiple files:
            dataset = Dataset(
                files="/dir/{year}/{month}/{day}/{hour}{minute}{second}.nc",
                name="TestData",
                # If the time coverage of the data cannot be retrieved from the
                # filename, you should set this to "content":
                time_coverage="filename"
            )

            # Find some files of the dataset:
            for file, times in dataset.find_files("2017-01-01", "2017-01-02"):
                # Should print some files such as "/dir/2017/01/01/120000.nc":
                print(file)

            ### Single file dataset ###
            # Define a dataset consisting of a single file:
            dataset = Dataset(
                # Simply use the files parameter without placeholders:
                files="/path/to/file.nc",
                name="TestData2",
                # The time coverage of the data cannot be retrieved from the
                # filename (because there are no placeholders). You can use the
                # file handler get_info() method via "content" or you can
                # define the time coverage here directly:
                time_coverage=("2007-01-01 13:00:00", "2007-01-14 13:00:00")
            )

            ### Play with the continuous flag ###
            # Define a dataset with daily files:
            dataset = Dataset("/dir/{year}/{month}/{day}.nc")

            times = dataset.retrieve_time_coverage(
                "/dir/2017/11/12.nc"
            )
            print("Start:", times[0])
            print("End:", times[1])

            # This prints actually:
            # Start: 2017-11-12
            # End: 2017-11-13
            # So the file covers a time period of one day although the end
            # time is not represented as placeholders in the files path.
            # The dataset interprets the file as continuous and
            # automatically sets its time coverage to the minimum resolution (
            # here one day). If you do not like this behaviour,
            # set Dataset.continuous to False.
            dataset.continuous = False

            times = dataset.retrieve_time_coverage(
                "/dir/2017/11/12.nc"
            )
            print("Start:", times[0])
            print("End:", times[1])
            # Start: 2017-11-12
            # End: 2017-11-12

        """

        # Initialize member variables:
        self._name = None
        self.name = name

        # The files parameters (will be set in the files setter method):
        self._files = None
        self.files_placeholders = None
        self.single_file = None
        self.files = files

        if handler is not None:
            self.handler = handler
        else:
            # Try to derive the file handler from the files extension:
            basename = self.files
            while True:
                basename, extension = os.path.splitext(basename)
                if not typhon.files.is_compression_format(
                        extension.lstrip(".")):
                    break

            if extension == ".nc" or extension == ".h5":
                self.handler = NetCDF4()
            elif extension == ".txt" or extension == ".asc" \
                    or extension == ".csv":
                self.handler = CSV()
            else:
                self.handler = None

        # Do the files cover everything (they are continuous) or are rather
        # single timestamps?
        self.continuous = continuous

        self.max_processes = max_processes
        self.compress = compress
        self.decompress = decompress

        if time_coverage is None:
            if self.single_file:
                # The default for single file datasets:
                self.time_coverage = [
                    datetime.min,
                    datetime.max
                ]
            else:
                # The default for multi file datasets:
                self.time_coverage = "filename"
        elif isinstance(time_coverage, (tuple, list)):
            if not self.single_file:
                warnings.warn(
                    "The explicit definition of the time coverage only makes "
                    "sense for single file datasets.", RuntimeWarning)

            self.time_coverage = [
                self._to_datetime(time_coverage[0]),
                self._to_datetime(time_coverage[1]),
            ]
        else:
            self.time_coverage = time_coverage

        # Multiple calls of .find_files() can be very slow when using a time
        # coverage retrieving method "content". Hence, we use a cache to
        # store the names and time coverages of already touched files in this
        # dictionary.
        self.times_cache_filename = times_cache
        self.time_coverages_cache = defaultdict(list)
        if self.times_cache_filename is not None:
            try:
                # Load the time coverages from a file:
                self.load_time_coverages(self.times_cache_filename)
            except Exception as e:
                raise e
            else:
                # Save the time coverages cache into a file before exiting.
                # This will be executed as well when the python code is
                # aborted due to an exception. This is normally okay, but what
                # happens if the error occurs during the loading of the time
                # coverages? We would overwrite the cache with nonsense.
                # Therefore, we need this code in this else block.
                atexit.register(
                    Dataset.save_time_coverages,
                    self, self.times_cache_filename)

    """def __iter__(self):
        return self

    def __next__(self):
        # We split the path of the input files after the first appearance of 
        # {day} or {doy}.
        path_parts = re.split(r'({\w+})', self.files)

        for dir in self._find_subdirs(path_parts[0])
            print(path_parts)

            yield file
    """
    def __contains__(self, item):
        """Checks whether a timestamp is covered by this dataset.

        Notes:
            This only gives proper results if the dataset consists of
            continuous data (files that covers a time span instead of only one
            timestamp).

        Args:
            item: Either a string with time information or datetime object.
                Can be also a tuple or list of strings / datetime objects that
                will be checked.

        Returns:
            True if timestamp is covered.
        """
        if isinstance(item, (tuple, list)):
            for elem in item:
                if elem not in self:
                    return False

            return True
        else:
            start = self._to_datetime(item)
            end = start + timedelta(microseconds=1)
            try:
                next(self.find_files(start, end, no_files_error=False,))
                return True
            except StopIteration:
                return False

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(self.read_period(item.start, item.stop))
        elif isinstance(item, (datetime, str)):
            filename = self.find_file(item)
            if filename is not None:
                return self.read(filename)
            return None

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start = key.start
            end = key.stop
        else:
            start = end = key

        filename = self.generate_filename(self.files, start, end)
        self.write(filename, value)

    def __repr__(self):
        return self.name

    def __str__(self):
        info = "Name:\t" + self.name
        info += "\nFiles:\t" + self.files
        return info

    def accumulate(self, start, end, concat_func=None, concat_args=None,
                   **reading_args):
        """Accumulate all data between two dates in one object.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            concat_func: Function that concatenates the read data to
                another. The first accepted argument must be a list of objects
                to concatenate. Default is ArrayGroup.concatenate.
            concat_args: A dictionary with additional arguments for
                *concat_func*.
            kwargs: A dictionary with additional arguments for reading
                the data (specified by the used file handler).

        Returns:
            Concatenated object.

        Examples:
            .. :code-block:: python

            import xarray

            dataset = Dataset("path/to/files.nc")
            data = dataset.accumulate(
                datetime(2016, 1, 1), datetime(2016, 2, 1),
                xarray.concat, read_args={fields : ("temperature", )})

            # do something with data["temperature"]
            data["temperature"].plot()
            ...
        """

        if concat_func is None:
            concat_func = ArrayGroup.concatenate

        if reading_args is None:
            contents = list(self.read_period(start, end))
        else:
            contents = list(self.read_period(start, end, **reading_args))

        if not contents:
            return None
        elif len(contents) == 1:
            return contents[0]

        if concat_args is None:
            return concat_func(contents)
        else:
            return concat_func(contents, **concat_args)

    @staticmethod
    def _call_function_with_file_info(args):
        """ This is a small wrapper function to call the function that is
        called on dataset files via .map().

        Args:
            args: A tuple containing following elements:
                (Dataset object, file_info, function,
                function_arguments)

        Returns:
            The return value of *function* called with the arguments *args*.
        """
        dataset, file_info, func, function_arguments, output, \
            return_file_info, verbose = args

        if verbose:
            print("Process %s ()" % file_info)

        if function_arguments is None:
            return_value = func(dataset, file_info)
        else:
            return_value = func(
                dataset, file_info, **function_arguments)

        if output is None:
            if return_file_info:
                return file_info, return_value
            else:
                return return_value
        else:
            # file_info could be a bundle of files
            if isinstance(file_info, list):
                start_times, end_times = zip(
                    *(
                        file.times
                        for file in file_info
                    )
                )
                output[min(start_times):max(end_times)] = return_value
            else:
                output[slice(file_info.times)] = return_value
            return file_info

    @staticmethod
    def _call_function_with_file_content(args):
        """ This is a small wrapper function to call a function on an object
        returned by reading a dataset file via Dataset.read().

        Args:
            args: A tuple containing following elements:
                (Dataset object, file_info, func,
                 function_arguments, read_arguments, output)

        Returns:
            The return value of *method* called with the arguments
            *method_arguments*.
        """
        dataset, file_info, func, function_arguments, output, \
            reading_arguments, return_file_info, verbose = args

        # file_info could be a bundle of files
        if isinstance(file_info, FileInfo):
            times = file_info.times
        else:
            start_times, end_times = zip(
                *(file.times for file in file_info)
            )
            times = min(start_times), max(end_times)

        if verbose:
            print("Process content from {} to {} ({} files)".format(
                *times,
                len(file_info) if isinstance(file_info, Iterable) else 1
            ))

        if reading_arguments is None:
            if isinstance(file_info, FileInfo):
                data = dataset.read(file_info)
            else:
                data = [dataset.read(file) for file in file_info]

        else:
            if isinstance(file_info, FileInfo):
                data = dataset.read(file_info, **reading_arguments)
            else:
                data = [
                    dataset.read(file, **reading_arguments)
                    for file in file_info
                ]

        if function_arguments is None:
            return_value = func(data)
        else:
            return_value = func(data, **function_arguments)

        if output is None:
            if return_file_info:
                return file_info, return_value
            else:
                return return_value
        elif return_value is not None:
            if verbose:
                print("\tSave output ...")
            output[slice(*times)] = return_value

        return file_info

    @staticmethod
    def _copy_file(
            dataset, filename, time_coverage,
            path, converter, delete_originals):
        """This is a small wrapper function for copying files. Do not use it
        directly but :meth:`Dataset.copy` instead.

        Args:
            dataset:
            filename:
            time_coverage:
            path:
            converter:
            delete_originals:

        Returns:
            None
        """
        # Generate the new file name
        new_filename = dataset.generate_filename(
            path, *time_coverage)

        # Create the new directory if necessary.
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)

        # Shall we simply copy or even convert the files?
        if converter is None:
            if delete_originals:
                print("\tDelete:", filename)
                shutil.move(filename, new_filename)
            else:
                shutil.copy(filename, new_filename)
        else:
            # Read the file with the current file handler
            data = dataset.read(filename)

            # Store the data of the file with the new file handler
            converter.write(new_filename, data)

            if delete_originals:
                print("\tDelete:", filename)
                os.remove(filename)

    def copy(
            self, start, end, destination,
            converter=None, joiner=None, delete_originals=False, verbose=False,
            new_name=None
    ):
        """ Copies all files from this dataset between two dates to another
        location.

        When passing a file handler via the argument converter, it also
        converts all matched files to a new format defined by the passed
        file handler.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            destination: The new path of the files. Must contain place holders
                (such as {year}, {month}, etc.).
            converter: If you want to convert the files during copying to a
                different format, you can pass a file handler object with
                writing-to-file support here.
            delete_originals: If true, then all copied original files will be
                deleted. Be careful, this cannot get undone!
            verbose: If true, it prints debug messages during copying.
            new_name: The name of the new dataset. If it is not given,
                the new name is the the old name followed by "_copy".

        Returns:
            New Dataset object with the new files.

        Examples:

        .. code-block:: python

            # Copy all the files between the 15th and 23rd September 2016:
            date1 = datetime(2017, 9, 15)
            date2 = datetime(2017, 9, 23)
            old_dataset = Dataset(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
                handler=FileHandlerJPG()
            )
            new_dataset = old_dataset.copy(
                date1, date2,
                "new/path/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
            )

        .. code-block:: python

            # When you want to convert the files during copying:
            old_dataset = Dataset(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
                handler=FileHandlerJPG()
            )
            # Note that this only works if the converter file handler
            # (FileHandlerPNG in this example) supports
            # writing to a file.
            new_dataset = old_dataset.copy(
                date1, date2,
                "new/path/{year}/{month}/{day}/{hour}{minute}{second}.png",
                converter=FileHandlerPNG(),
            )
        """

        # If the new path contains place holders, fill them for each file.
        # Otherwise it is a path which does not describe each file
        # individually. So far, we cannot handle this.
        # TODO: Adjust this solution for single file datasets.
        # TODO: Is it helpful for the performance to use multiple processes
        # TODO: here?
        if "{" in destination:
            # Copy the files
            self.map(
                start, end, Dataset._copy_file,
                 {
                     "path" : destination,
                     "converter" : converter,
                     "delete_originals" : delete_originals
                 },
                 verbose=verbose
            )
        else:
            if self.single_file:
                # TODO: Copy single file
                raise NotImplementedError("Copying single files is not yet "
                                          "implemented!")
            else:
                raise ValueError(
                    "The new_path argument must describe each file "
                    "individually by using place holders!")

        # Copy this dataset object but change the files parameter.
        new_dataset = Dataset(
            destination,
            new_name if new_name is not None else self.name + "_copy",
            self.handler, self.time_coverage
        )

        if converter is not None:
            # The files are in a different format now. Hence, we need the new
            # file handler:
            new_dataset.handler = converter

        return new_dataset

    @staticmethod
    def _create_date_from_placeholders(placeholders, values, prefix=None,
                                       exclude=None, default=None):
        """Creates a dictionary with date and time keys from placeholders and
        their values.

        Args:
            placeholders:
            values:
            prefix:
            exclude:
            default:

        Returns:
            A dictionary with "year", "month", etc.
        """
        date_args = {}

        if prefix is None:
            prefix = ""

        if default is not None:
            date_args.update(**default)

        for index, placeholder in enumerate(placeholders):
            value = int(values[index])
            if placeholder == prefix + "year2":
                # TODO: What should be the threshold that decides whether the
                # TODO: year is 19xx or 20xx?
                if value < 65:
                    date_args["year"] = 2000 + value
                else:
                    date_args["year"] = 1900 + value
            elif placeholder == prefix + "millisecond":
                date_args[prefix + "microsecond"] = value * 1000
            elif (exclude is None or not placeholder.startswith(exclude)) \
                    and placeholder.startswith(prefix):
                if prefix is None:
                    date_args[placeholder] = value
                else:
                    # Cut off the prefix
                    date_args[placeholder[len(prefix):]] = value

        if prefix + "doy" in placeholders:
            date = datetime(date_args["year"], 1, 1) \
                   + timedelta(date_args["doy"] - 1)
            date_args["month"] = date.month
            date_args["day"] = date.day
            del date_args["doy"]

        return date_args

    @property
    def files(self):
        """Gets or sets the path to the dataset's files.

        Returns:
            A string with the path (can contain placeholders or wildcards.)
        """
        return self._files

    @files.setter
    def files(self, value):
        if value is None:
            raise ValueError("The files parameter cannot be None!")
        self._files = value

        self.files_placeholders = re.findall("\{(\w+)\}", self.files)

        # Flag whether this is a single file dataset or not:
        self.single_file = \
            not self.files_placeholders \
            and "*" not in self.files

    def find_file(self, timestamp):
        """Finds either the file that covers a timestamp or is the closest to
        it.

        TODO: Make clear what this method returns.

        Notes:
            It is guaranteed that this method returns a filename. That does not
            mean that the file is close to the timestamp. It is simply the
            closest.

        Args:
            timestamp: date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.

        Returns:
            The found file name as a string. If no file was found, a
            NoFilesError is raised.
        """

        # Special case: the whole dataset consists of one file only.
        if self.single_file:
            if os.path.isfile(self.files):
                # We do not have to check the time coverage since there this is
                # automatically the closest file to the timestamp.
                return self.files
            else:
                raise ValueError(
                    "The files parameter of '%s' does not contain placeholders"
                    " and is not a path to an existing file!" % self.name)

        timestamp = self._to_datetime(timestamp)

        # Maybe there is a file with exact this timestamp?
        path = self.generate_filename(
            self.files, timestamp, timestamp
        )

        if os.path.exists(path):
            return path

        # We need all possible files that are close to the timestamp hence we
        # need the search dir for those files:
        search_dir = self.generate_filename(
            os.path.dirname(self.files), timestamp
        )

        # Prepare the regex:
        regex = self.files.format(**Dataset.placeholder)
        regex = re.compile(regex.replace("*", ".*?"))
        files = list(self._get_files(
            search_dir, regex, datetime.min, datetime.max, False
        ))

        if not files:
            return None

        times = [file.times for file in files]

        # Either we find a file that covers the certain timestamp:
        for index, time_coverage in enumerate(times):
            if IntervalTree.contains(time_coverage, timestamp):
                return files[index]

        # Or we find the closest file.
        intervals = np.min(np.abs(np.asarray(times) - timestamp), axis=1)
        return files[np.argmin(intervals)]

    def find_files(
            self, start, end, sort=True, bundle=None,
            no_files_error=True, verbose=False,
    ):
        """ Find all files of this dataset in a given time period.

        The *start* and *end* parameters build a semi-open interval: only the
        files that are equal or newer than *start* and older than *end* are
        going to be found.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sort: If true, all files will be yielded
                sorted by their starting time. Default is true.
            bundle: Instead of only yielding one file at a time, you can get a
                bundle of files. There are two possibilities: by setting this
                to an integer, you can define the size of the bundle directly
                or by setting this to a string (e.g. *1H*),
                you can define the time period of one bundle. See
                http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
                for allowed time specifications. Default value is 1. This
                argument will be ignored when having a single-file dataset.
                When using *bundle*, the returned files will always be sorted
                ignoring the state of the *sort* argument.
            no_files_error: If true, raises an NoFilesError when no
                files are found.
            verbose: If true, debug messages will be printed.

        Yields:
            Either a :class:`~typhon.spareice.handlers.FileInfo` object for
            each found file or - if *bundle_size* is not None - a list of
            :class:`~typhon.spareice.handlers.FileInfo` objects.

        Examples:

        .. code-block:: python

            # Define a dataset consisting of multiple files:
            dataset = Dataset(
                "/dir/{year}/{month}/{day}/{hour}{minute}{second}.nc"
            )

            # Find some files of the dataset:
            for file in dataset.find_files("2017-01-01", "2017-01-02"):
                # file is a FileInfo object that has the attribute path
                # and times.
                print(file.path)  # e.g. "/dir/2017/01/01/120000.nc"
                print(file.times)  # list of two datetime objects
        """

        # The user can give strings instead of datetime objects:
        start = self._to_datetime(start)
        end = self._to_datetime(end)

        if verbose:
            print("Find files between %s and %s!" % (start, end))

        # Special case: the whole dataset consists of one file only.
        if self.single_file:
            if os.path.isfile(self.files):
                time_coverage = self.retrieve_time_coverage(self.files)
                if IntervalTree.overlaps(time_coverage, (start, end)):
                    yield FileInfo(self.files, time_coverage)
                elif no_files_error:
                    raise NoFilesError(self.name, start, end)
                return
            else:
                raise ValueError(
                    "The files parameter of '%s' neither contains placeholders"
                    " nor is a path to an existing file!" % self.name)

        # Prepare the regex for the file path
        regex = self.files.format(**Dataset.placeholder)
        regex = re.compile(regex.replace("*", ".*?"))

        # We want to have a semi-open interval as explained in the doc string.
        end -= timedelta(microseconds=1)

        # Find all files by iterating over all searching paths and check
        # whether they match the path regex and the time period.
        file_finder = (
            file_info
            for path in self._get_search_dirs(start, end, verbose)
            for file_info in self._get_files(path, regex, start, end, verbose)
        )

        # Even if no files were found, the user does not want to know.
        if not no_files_error:
            yield from self._prepare_return_value(file_finder, sort, bundle)
            return

        # The users wants an error to be raised if no files were found. Since
        # the file_finder is an iterator, we have to check whether it is empty.
        # I do not know whether there is a more pythonic way but Matthew
        # Flaschen shows how to do it with itertools.tee:
        # https://stackoverflow.com/a/3114423
        return_files, check_files = tee(file_finder)
        try:
            next(check_files)

            # We have found some files and can return them
            yield from self._prepare_return_value(return_files, sort, bundle)
        except StopIteration:
            raise NoFilesError(self.name, start, end)

    def _get_search_dirs(self, start, end, verbose):
        """Yields the search directory for a time period.

        Args:
            start: Datetime that defines the start of a time interval.
            end: Datetime that defines the end of a time interval. The time
                coverage of the files should overlap with this interval.
            verbose: If true, it prints debug messages.

        Yields:
            A path as a string.
        """

        dir_template = os.path.dirname(self.files)

        # We need the minimum resolution of the temporal placeholders in the
        # directory name to create adequate search paths.
        placeholders = set(re.findall("\{(\w+)\}", dir_template))
        temporal_placeholders = placeholders.intersection(
            Dataset._temporal_resolution.keys())

        if verbose and temporal_placeholders:
            print("Found temporal placeholders: ", temporal_placeholders)

        # If the directory does not contain temporal placeholders, we simply
        # return the original directory
        if not temporal_placeholders:
            if verbose:
                print("Found no temporal placeholders")
            yield dir_template
            return

        # From all temporal placeholders, we want to find the one with the
        # lowest resolution (month > day > hour, etc.).
        # Note: The higher the resolution of the placeholder is the lower its
        # sorting rank is.
        temporal_resolution = max(
            [Dataset._temporal_resolution[tp] for tp in temporal_placeholders],
            key=lambda x: x[1]
        )[0]

        if verbose:
            print("Temporal resolution of search path:", temporal_resolution)

        # Start one day before the starting date because we may have files
        # overlapping one day.
        times = pd.date_range(
            start.date() - timedelta(days=1), end,
            freq=temporal_resolution,
        )

        if verbose:
            print("Searching between %s and %s in approximately %d dirs" % (
                times[0], times[-1], len(times)
            ))

        for dir_time in times:
            search_dir = self.generate_filename(dir_template, dir_time)

            if verbose:
                print("Search directory:", search_dir)

            if not os.path.isdir(search_dir):
                if verbose:
                    print("\tSkipped -> is no directory!")
                continue

            yield search_dir

        return

    def _get_files(self, path, regex, start, end, verbose):
        """Yield files that matches the search conditions.

        Args:
            path: Path to the directory that contains the files that should be
                checked.
            regex: A regular expression that should match the filename.
            start: Datetime that defines the start of a time interval.
            end: Datetime that defines the end of a time interval. The time
                coverage of the file should overlap with this interval.
            verbose: If True, it prints debug messages.

        Yields:
            A FileInfo object with the file path and time coverage
        """

        if verbose:
            print("Check all files in %s" % os.path.join(path, "*"))

        for filename in glob.iglob(os.path.join(path, "*")):
            if regex.match(filename):
                if verbose:
                    print(filename)

                times = self.retrieve_time_coverage(filename)

                # Test whether the file is overlapping the interval between
                # start and end date.
                if IntervalTree.overlaps(times, (start, end)):
                    if verbose:
                        print("\tPassed time check")
                    yield FileInfo(filename, times)
                elif verbose:
                    print("\tFailed time check")

    @staticmethod
    def _get_time_resolution(date_args):
        if "millisecond" in date_args:
            return timedelta(milliseconds=1)
        elif "second" in date_args:
            return timedelta(seconds=1)
        elif "minute" in date_args:
            return timedelta(minutes=1)
        elif "hour" in date_args:
            return timedelta(hours=1)
        elif "day" in date_args:
            return timedelta(days=1)
        else:
            return None

    @staticmethod
    def _prepare_return_value(file_iterator, sort, bundle_size):
        """Prepares the return value of the find_files method.

        Args:
            file_iterator: Generator function that yields the found files.
            sort: If true, all found files will be sorted according to their
                starting times.
            bundle_size: See the documentation of the *bundle* argument in
                :meth`find_files` method.

        Yields:
            Either one FileInfo object or - if bundle_size is set - a list of
            FileInfo objects.
        """
        # We want to have sorted files if we want to bundle them.
        if sort or isinstance(bundle_size, int):
            file_iterator = sorted(file_iterator, key=lambda x: x.times[0])

        if bundle_size is None:
            yield from file_iterator
            return

        # The argument bundle was defined. Either it sets the bundle size
        # directly via a number or indirectly by setting time periods.
        if isinstance(bundle_size, int):
            files = list(file_iterator)

            yield from (
                files[i:i+bundle_size]
                for i in range(0, len(files), bundle_size)
            )
        elif isinstance(bundle_size, str):
            files = list(file_iterator)

            # We want to split the files into hourly (or daily, etc.) bundles.
            # pandas provides a practical grouping function.
            time_series = pd.Series(
                files,
                [file.times[0] for file in files]
            )
            yield from (
                bundle[1].values.tolist()
                for bundle in time_series.groupby(
                    pd.TimeGrouper(freq=bundle_size))
                if bundle[1].any()
            )
        else:
            raise ValueError(
                "The parameter bundle must be a integer or string!")

    def find_overlapping_files(
            self, start, end, other_dataset, max_interval=None, verbose=False):
        """Find files between two datasets that overlap in time.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            other_dataset: A Dataset object which holds the other files.
            max_interval: Maximal time interval in seconds between
                two overlapping files. Must be an integer or float.
            verbose: If true, debugging messages will be printed.

        Yields:
            A tuple with the names of two files which correspond to each other.
        """
        if max_interval is not None:
            max_interval = self._to_timedelta(max_interval)
            start = self._to_datetime(start) - max_interval
            end = self._to_datetime(end) + max_interval

        files1 = list(
            self.find_files(start, end, verbose=verbose)
        )
        files2 = list(
            other_dataset.find_files(start, end, verbose=verbose)
        )

        # Convert the times (datetime objects) to seconds (integer)
        times1 = [
            [int(file.times[0].timestamp()), int(file.times[1].timestamp())]
            for file in files1
        ]
        times2 = np.asarray([
            [file.times[0].timestamp(), file.times[1].timestamp()]
            for file in files2
        ]).astype('int')

        if max_interval is not None:
            # Expand the intervals of the secondary dataset to close-in-time
            # intervals.
            times2[:, 0] -= int(max_interval.total_seconds())
            times2[:, 1] += int(max_interval.total_seconds())

        tree = IntervalTree(times2)

        # Search for all overlapping intervals:
        results = tree.query(times1)

        yield from (
            (files1[i],
             [files2[oi] for oi in sorted(overlapping_files)])
            for i, overlapping_files in enumerate(results)
        )

    def generate_filename(self, template, start_time, end_time=None):
        """ Generates the file name for a specific date by using the template
        argument.

        Allowed placeholders in the template are:

        +-------------+------------------------------------------+------------+
        | Placeholder | Description                              | Example    |
        +=============+==========================================+============+
        | year        | Four digits indicating the year.         | 1999       |
        +-------------+------------------------------------------+------------+
        | year2       | Two digits indicating the year. [1]_     | 58 (=2058) |
        +-------------+------------------------------------------+------------+
        | month       | Two digits indicating the month.         | 09         |
        +-------------+------------------------------------------+------------+
        | day         | Two digits indicating the day.           | 08         |
        +-------------+------------------------------------------+------------+
        | doy         | Three digits indicating the day of       | 002        |
        |             | the year.                                |            |
        +-------------+------------------------------------------+------------+
        | hour        | Two digits indicating the hour.          | 22         |
        +-------------+------------------------------------------+------------+
        | minute      | Two digits indicating the minute.        | 58         |
        +-------------+------------------------------------------+------------+
        | second      | Two digits indicating the second.        | 58         |
        +-------------+------------------------------------------+------------+
        | millisecond | Three digits indicating the millisecond. | 999        |
        +-------------+------------------------------------------+------------+
        .. [1] Numbers lower than 65 are interpreted as 20XX while numbers
            equal or greater are interpreted as 19XX (e.g. 65 = 1965,
            99 = 1999)

        All those place holders are also allowed to have the prefix *end*
        (e.g. *end_year*). They represent the end of the time coverage.

        Args:
            template: A string with format placeholders such as {year} or
                {day}.
            start_time: A datetime object with the needed date and time.
            end_time: A datetime object. All placeholders with the
                prefix *end* will be filled with this datetime. If this is not
                given, it will be set to *start_time*.

        Returns:
            A string containing the full path and name of the file.

        Example:

        .. code-block:: python

            Dataset.generate_filename(
                "{year2}/{month}/{day}.dat",
                datetime(2016, 1, 1)
            )
            # Returns "16/01/01.dat"

            Dataset.generate_filename(
                "{year}{month}{day}-{end_year}{end_month}{end_day}.dat",
                datetime(2016, 1, 1),
                datetime(2016, 12, 31),
            )
            # Returns "20160101-20161231.dat"

        """

        start_time = Dataset._to_datetime(start_time)

        if end_time is None:
            end_time = start_time
        else:
            end_time = Dataset._to_datetime(end_time)

        # Fill all placeholders variables with values
        template = template.format(
            year=start_time.year, year2=str(start_time.year)[-2:],
            month="{:02d}".format(start_time.month),
            day="{:02d}".format(start_time.day),
            doy="{:03d}".format(
                (start_time - datetime(start_time.year, 1, 1)).days
                + 1),
            hour="{:02d}".format(start_time.hour),
            minute="{:02d}".format(start_time.minute),
            second="{:02d}".format(start_time.second),
            millisecond="{:03d}".format(int(start_time.microsecond / 1000)),
            end_year=end_time.year, end_year2=str(end_time.year)[-2:],
            end_month="{:02d}".format(end_time.month),
            end_day="{:02d}".format(end_time.day),
            end_doy="{:03d}".format(
                (end_time - datetime(end_time.year, 1, 1)).days
                + 1),
            end_hour="{:02d}".format(end_time.hour),
            end_minute="{:02d}".format(end_time.minute),
            end_second="{:02d}".format(end_time.second),
            end_millisecond="{:03d}".format(int(end_time.microsecond / 1000)),
            name=self.name,
        )

        return template

    def get_info(self, filename):
        """Gets info about a file by using the dataset's file handler.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: Path and name of the file.

        Returns:
            Dictionary with information about the file.
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not get info from the file '{}'! No file handler is "
                "specified!".format(filename))

        with typhon.files.decompress(filename) as file:
            return self.handler.get_info(file)

    def load_time_coverages(self, filename):
        """ Loads the time coverages cache from a file.

        Returns:
            None
        """
        if filename is not None and os.path.exists(filename):
            print("Load time coverages of {} dataset from {}.".format(
                self.name, filename))

            try:
                with open(filename) as file:
                    time_coverages = json.load(file)
                    # Parse string times into datetime objects:
                    time_coverages = {
                        file: [
                           datetime.strptime(times[0], "%Y-%m-%dT%H:%M:%S.%f"),
                           datetime.strptime(times[1], "%Y-%m-%dT%H:%M:%S.%f")
                        ]
                        for file, times in time_coverages.items()
                    }
                    self.time_coverages_cache.update(time_coverages)
            except json.decoder.JSONDecodeError as e:
                warnings.warn(
                    "Could not load the time coverages from cache file '%s':\n"
                    "%s." % (filename, e)
                )

    def map(
        self, start, end,
        func, func_arguments=None, output=None, max_processes=None,
        bundle=None, return_file_info=False, process_initializer=None,
        process_initargs=None, verbose=False,
    ):
        """Applies a function on all files of this dataset between two dates.

        This method can use multiple processes to boost the procedure
        significantly. Depending on which system you work, you should try
        different numbers for *max_processes*.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            func: A reference to a function. The function should accept
                at least three arguments: the dataset object, the filename and
                the time coverage of the file (tuple of two datetime objects).
            func_arguments: Additional keyword arguments for the function.
            output: Set this to a Dataset object and the return value of
                *func* will be copied there. In that case
                *include_file_info* will be ignored.
            max_processes: Max. number of parallel processes to use. When
                lacking performance, you should change this number.
            bundle: Instead of only mapping a function onto one file at a time,
                you can map it onto a bundle of files. Look at the
                documentation of the *bundle* argument in
                :meth:`~typhon.spareice.Dataset.find_files` for more details.
            return_file_info: Since the order of the returning results is
                arbitrary, you can include the name of the processed file
                and its time coverage in the results.
            process_initializer: Must be a reference to a function that is
                called once when starting a new process. Can be used to preload
                variables into one process workspace. See also
                https://docs.python.org/3.1/library/multiprocessing.html#module-multiprocessing.pool
                for more information.
            process_initargs: A tuple with arguments for *process_initializer*.
            verbose: If this is true, debug information will be printed.

        Returns:
            A list with one item for each processed file. The order is
            arbitrary. If *return_file_info* is true, the item is a tuple
            of a FileInfo object and the return value of the applied function.
            If *return_file_info* is false, the item is simply the return
            value of the applied function.

        Examples:

        """

        if verbose:
            print("Process all files from %s to %s.\nThis may take a while..."
                  % (start, end))

        # Measure the time for profiling.
        start_time = time.time()

        if max_processes is None:
            max_processes = self.max_processes

        # Create a pool of processes and process all the files with them.
        pool = Pool(
            max_processes, initializer=process_initializer,
            initargs=process_initargs,
        )

        args = (
            (self, x, func, func_arguments, output, return_file_info, verbose)
            for x in self.find_files(start, end, sort=False, bundle=bundle, )
        )

        results = pool.map(
            Dataset._call_function_with_file_info,
            args, #chunksize=10,
        )

        if verbose:
                print("It took %.2f seconds using %d parallel processes to "
                      "process %d files." % (
                        time.time() - start_time, max_processes, len(results)))

        return results

    def map_content(
            self, start, end,
            func, func_arguments=None, output=None, reading_arguments=None,
            max_processes=None, bundle=None, return_file_info=False,
            process_initializer=None, process_initargs=None,
            verbose=False):
        """Applies a method on the content of each file of this dataset between
        two dates.

        This method is similar to Dataset.map() but each file will be read
        before the given function will be applied.

        This method can use multiple processes to boost the procedure
        significantly. Depending on which system you work, you should try
        different numbers for *max_processes*.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            func: A reference to a function. The function should expect as
                first argument the content object which is returned by the file
                handler's *read* method.
            func_arguments: Additional keyword arguments for the function.
            reading_arguments: Additional keyword arguments that will be passed
                to the reading function (see Dataset.read() for more
                information).
            output: Set this to a Dataset object and the return value of
                *func* will be copied there. In that case
                *return_file_info* will be ignored.
            max_processes: Max. number of parallel processes to use. When
                lacking performance, you should change this number.
            return_file_info: Since the order of the returning results is
                arbitrary, you can include the name of the processed file
                and its time coverage in the results.
            process_initializer: Must be a reference to a function that is
                called once when starting a new process. Can be used to preload
                variables into one process workspace. See also
                https://docs.python.org/3.1/library/multiprocessing.html#module-multiprocessing.pool
                for more information.
            process_initargs: A tuple with arguments for *process_initializer*.
            bundle: Instead of only mapping a function onto one file at a time,
                you can map it onto a bundle of files. Look at the
                documentation of the *bundle* argument in
                :meth:`~typhon.spareice.Dataset.find_files` for more details.
            verbose: If  true, debug information will be printed.

        Returns:
            A list with one item for each processed file. The order is
            arbitrary.
            If *output* is set to a Dataset object, only a list with all
                processed files is returned.
            If *return_file_info* is true, the item is a tuple of a FileInfo
                object and the return value of the applied function.
            If *return_file_info* is false, it is simply the return value
                of the applied function.

        Examples:

        """

        if verbose:
            print("Process all files from %s to %s.\nThis may take a while..."
                  % (start, end))

            # Measure the time for profiling.
            start_time = time.time()

        if max_processes is None:
            max_processes = self.max_processes

        # Create a pool of processes and process all the files with them.
        pool = Pool(
            max_processes, initializer=process_initializer,
            initargs=process_initargs,
        )

        # Prepare argument list for mapping function
        args = (
            (self, x, func, func_arguments, output, reading_arguments,
             return_file_info, verbose)
            for x in self.find_files(start, end, sort=False, bundle=bundle, )
        )

        results = pool.map(
            Dataset._call_function_with_file_content,
            args, #chunksize=10,
        )

        if verbose:
            print("It took %.2f seconds using %d parallel processes to "
                  "process %d files." % (
                    time.time() - start_time, max_processes, len(results)))

        return results

    @property
    def name(self):
        """Gets or sets the dataset's name.

        Returns:
            A string with the dataset's name.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            value = str(id(self))

        self._name = value
        self.placeholder["name"] = value

    def read(self, filename, **reading_arguments):
        """Opens and reads a file.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: A string, path-alike object or an iterable of
            **reading_arguments: Additional key word arguments for the
                *read* method of the used file handler class.

        Returns:
            The content of the read file.
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not get read the file '{}'! No file handler is "
                "specified!".format(filename))

        if isinstance(filename, FileInfo):
            filename = filename.path

        if self.decompress:
            with typhon.files.decompress(filename) as file:
                return self.handler.read(file, **reading_arguments)
        else:
            return self.handler.read(filename, **reading_arguments)

    def read_period(self, start, end, sort=True, **reading_arguments):
        """Reads all files between two dates and returns their content sorted
        by their starting time.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sort: Sort the files by their starting time.
            **reading_arguments: Additional key word arguments for the
                *read* method of the used file handler class.

        Yields:
            The content of the read file.
        """
        #
        for file in self.find_files(start, end, sort=sort):
            data = self.read(file, **reading_arguments)
            if data:
                yield data

    def retrieve_time_coverage(self, filename, retrieve_method=None):
        """ Retrieves the time coverage from a given dataset file.

        Args:
            filename: Name of the file as string.
            retrieve_method: Defines how the time coverage should be
                retrieved. If you set this to *filename* (default) then the
                time coverage is retrieved from the filename by using the
                placeholders. This is fast but could lead to errors due to
                ambiguity. This only works if placeholders have been used in
                the *files* parameter of the Dataset object. Look at the
                documentation of the :meth:`generate_filename` method to get an
                overview about the allowed placeholders. If this parameter is
                *content* then the :meth:`get_info` method is used to determine
                the time coverage. This normally means that the file is opened
                and its content is checked. This could be more time consuming
                but also more reliable.

        Returns:
            A tuple of two datetime objects, indicating the time coverage.
        """

        # We use a cache for time coverages, this makes everything faster:
        if filename in self.time_coverages_cache:
            return self.time_coverages_cache[filename]

        if retrieve_method is None:
            if self.time_coverage != "filename" \
                    and self.time_coverage != "content":
                return self.time_coverage
            retrieve_method = self.time_coverage

        if retrieve_method == "filename":
            if self.single_file:
                raise ValueError(
                    "The files parameter does not contain any placeholders! I "
                    "could not retrieve the time coverage from it."
                )

            regex = self.files.format(**Dataset.placeholder)
            regex = re.compile(regex.replace("*", ".*?"))
            try:
                values = regex.findall(filename)
                values = values[0]
            except IndexError:
                raise ValueError(
                    "The filename does not match the given template from the "
                    "parameter 'files'. I could not retrieve the time "
                    "coverage.")

            start_date_args = self._create_date_from_placeholders(
                self.files_placeholders, values, exclude="end_")

            # Default: if no end date is given then the starting date is also
            # the end date.

            end_date_args = self._create_date_from_placeholders(
                self.files_placeholders, values, prefix="end_",
                default=start_date_args)

            start_date = datetime(**start_date_args)
            end_date = datetime(**end_date_args)

            # Automatically extend the coverage for the minimal resolution
            # of the retrieved datetime objects if there is no end date.
            if self.continuous and start_date == end_date:
                end_date += self._get_time_resolution(start_date_args)
                end_date -= timedelta(microseconds=1)

            # Sometimes the filename does not explicitly provide the complete
            # end date. Imagine there is only hour and minute given, then day
            # change would not be noticed. Therefore, make sure that the end
            # date is always bigger (later) than the start date.
            if end_date < start_date:
                end_date += timedelta(days=1)

            time_coverage = (start_date, end_date)
        elif retrieve_method == "content":
            time_coverage = self.get_info(filename).times

        # Cache the time coverage of this file:
        self.time_coverages_cache[filename] = time_coverage

        return time_coverage

    def save_time_coverages(self, filename):
        """ Saves time coverages cache to a file.

        Returns:
            None
        """
        if filename is not None:
            print("Save time coverages of {} dataset to {}.".format(
                self.name, filename))
            with open(filename, 'w') as file:
                # We cannot save datetime objects with json directly. We have
                # to convert them to strings first:
                time_coverages = {
                    file: (
                        times[0].strftime("%Y-%m-%dT%H:%M:%S.%f"),
                        times[1].strftime("%Y-%m-%dT%H:%M:%S.%f"))
                    for file, times in self.time_coverages_cache.items()
                }
                json.dump(time_coverages, file)

    @staticmethod
    def _to_datetime(obj):
        if isinstance(obj, str):
            return pd.Timestamp(obj).to_pydatetime()
        elif isinstance(obj, datetime):
            return obj
        else:
            raise KeyError("Cannot convert object of type '%s' to datetime "
                           "object! Allowed are only datetime or string "
                           "objects!" % type(obj))

    @staticmethod
    def _to_timedelta(obj):
        if isinstance(obj, numbers.Number):
            return timedelta(seconds=obj)
        elif isinstance(obj, timedelta):
            return obj
        else:
            raise KeyError("Cannot convert object of type '%s' to timedelta"
                           "object! Allowed are only timedelta or number "
                           "objects!" % type(obj))

    def write(self, filename, data, **writing_arguments):
        """Writes content to a file by using the Dataset's file handler.

        If the filename extension is a compression format (such as *zip*,
        etc. look at :func:`typhon.files.is_compression_format` for a list) and
        *Dataset.compress* is set to true, the file will be compressed.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: Path and name of the file where to put the data.
            data: An object that can be stored by the used file handler class.
            **writing_arguments: Additional key word arguments for the
            *write* method of the used file handler class.

        Returns:
            None
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not write data to the file '{}'! No file handler is "
                "specified!".format(filename))

        if isinstance(filename, FileInfo):
            filename = filename.path

        # The users should not be bothered with creating directories by
        # themselves.
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if self.compress:
            with typhon.files.compress(filename) as file:
                return self.handler.write(file, data, **writing_arguments)
        else:
            return self.handler.write(filename, data, **writing_arguments)


class DatasetManager(dict):
    def __init__(self, *args, **kwargs):
        """ This manager can hold multiple Dataset objects. You can use it as a
        native dictionary.

        More functionality will be added in future.

        Example:

        .. code-block:: python

            datasets = DatasetManager()

            datasets += Dataset(
                name="dumbo.thermocam",
                files="path/to/files",
                handler=cloud.handler.dumbo.ThermoCamFile())

            ## do something with it
            for name, dataset in datasets.items():
                dataset.find_files(...)

        """
        super(DatasetManager, self).__init__(*args, **kwargs)

    def __iadd__(self, dataset):
        if dataset.name in self:
            warnings.warn(
                "DatasetManager: Overwrite dataset with name '%s'!"
                % dataset.name, RuntimeWarning)

        self[dataset.name] = dataset
        return self
