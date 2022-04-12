#!/usr/bin/env python
import sys
import json
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
from config import Config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = Config()


class db(object):
    def __init__(self, config):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = 'db/db_schema.txt'
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        """Enter method."""
        if main_config.db_ssh_forward:
            forward = sshtunnel.SSHTunnelForwarder(
                credentials.machine_credentials()['ssh_address'],
                ssh_username=credentials.machine_credentials()['username'],
                ssh_password=credentials.machine_credentials()['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            self.forward = forward
            self.pgsql_port = forward.local_bind_port
        else:
            self.forward = None
            self.pgsql_port = ''
        pgsql_string = credentials.postgresql_connection(str(self.pgsql_port))
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def recreate_db(self):
        """Initialize the DB from the schema file."""
        db_schema = open(self.db_schema_file).read().splitlines()
        for s in db_schema:
            t = s.strip()
            if len(t):
                self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                )

    def populate_db(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO experiments
            (
                experiment,
                model,
                train_dataset,
                val_dataset,
                validation_period,
                validation_steps,
                shuffle_val,
                shuffle_train,
                lr,
                train_loss_function,
                val_loss_function,
                score_function,
                optimizer,
                train_batch_size,
                val_batch_size,
                epochs,
                get_map,
                save_checkpoints,
                save_activities,
                save_weights,
                save_gradients,
                train_augmentations,
                val_augmentations
            )
            VALUES
            (
                %(experiment)s,
                %(model)s,
                %(train_dataset)s,
                %(val_dataset)s,
                %(validation_period)s,
                %(validation_steps)s,
                %(shuffle_val)s,
                %(shuffle_train)s,
                %(lr)s,
                %(train_loss_function)s,
                %(val_loss_function)s,
                %(score_function)s,
                %(optimizer)s,
                %(train_batch_size)s,
                %(val_batch_size)s,
                %(epochs)s,
                %(get_map)s,
                %(save_checkpoints)s,
                %(save_activities)s,
                %(save_weights)s,
                %(save_gradients)s,
                %(train_augmentations)s,
                %(val_augmentations)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_parameters_and_reserve(self, experiment=None, random=True):
        """Pull parameters and update the in process table."""
        if experiment is not None:
            exp_string = """experiment='%s' and""" % experiment
        else:
            exp_string = """"""
        if random:
            rand_string = """ORDER BY random()"""
        else:
            rand_string = """"""
        self.cur.execute(
            """
            INSERT INTO in_process (experiment_id, experiment)
            (SELECT _id, experiment FROM experiments h
            WHERE %s NOT EXISTS (
                SELECT 1
                FROM in_process i
                WHERE h._id = i.experiment_id
                )
            %s LIMIT 1)
            RETURNING experiment_id
            """ % (
                exp_string,
                rand_string,
            )
        )
        self.cur.execute(
            """
            SELECT * FROM experiments
            WHERE _id=%(_id)s
            """,
            {
                '_id': self.cur.fetchone()['experiment_id']
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_parameters_for_evaluation(
            self,
            experiment_name=None,
            random=True,
            ckpt_path=None):
        """Pull parameters without updating the in process table."""
        self.cur.execute(
            """
            SELECT * FROM performance
            WHERE ckpt_path=%(ckpt_path)s
            """,
            {
                'ckpt_path': ckpt_path
            }
        )
        self.cur.execute(
            """
            SELECT * FROM experiments
            WHERE _id=%(_id)s
            """,
            {
                '_id': self.cur.fetchone()['experiment_id']
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def list_experiments(self):
        """List all experiments."""
        self.cur.execute(
            """
            SELECT distinct(experiment) from experiments
            """
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def update_in_process(self, experiment_id, experiment):
        """Update the in_process table."""
        self.cur.execute(
            """
             INSERT INTO in_process
             VALUES
             (%(experiment_id)s, %(experiment_name)s)
            """,
            {
                'experiment_id': experiment_id,
                'experiment': experiment
            }
        )
        if self.status_message:
            self.return_status('INSERT')

    def get_performance(self, experiment_name):
        """Get experiment performance."""
        if len(experiment_name.split(',')) > 1:
            experiment_name = experiment_name.replace(',', '|')
            experiment_name = '(%s)' % experiment_name
        elif '|' in experiment_name:
            experiment_name = '(%s)' % experiment_name
        self.cur.execute(
            """
            SELECT * FROM performance AS P
            INNER JOIN experiments ON experiments._id = P.experiment_id
            WHERE experiments.experiment ~ %(experiment_name)s
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def remove_experiment(self, experiment):
        """Delete an experiment from all tables."""
        self.cur.execute(
            """
            DELETE FROM experiments WHERE experiment=%(experiment)s;
            DELETE FROM performance WHERE experiment_name=%(experiment)s;
            DELETE FROM in_process WHERE experiment_name=%(experiment)s;
            """,
            {
                'experiment': experiment
            }
        )
        if self.status_message:
            self.return_status('DELETE')

    def reset_in_process(self):
        """Reset in process table."""
        self.cur.execute(
            """
            DELETE FROM in_process
            """
        )
        if self.status_message:
            self.return_status('DELETE')

    def update_performance(self, namedict):
        """Update performance in database."""
        self.cur.execute(
            """
            INSERT INTO performance
            (
            experiment_id,
            experiment,
            train_score,
            train_loss,
            val_score,
            val_loss,
            step,
            ckpt_path,
            summary_path,
            num_params,
            results_path
            )
            VALUES
            (
            %(experiment_id)s,
            %(experiment)s,
            %(train_score)s,
            %(train_loss)s,
            %(val_score)s,
            %(val_loss)s,
            %(step)s,
            %(ckpt_path)s,
            %(summary_path)s,
            %(num_params)s,
            %(results_path)s
            )
            RETURNING _id""",
            namedict
        )
        if self.status_message:
            self.return_status('SELECT')

    def get_report(self, experiment_name):
        """Get experiment performance."""
        self.cur.execute(
            """
            SELECT * FROM performance AS P
            LEFT JOIN experiments ON experiments._id = P.experiment_id
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()


def get_experiment_name():
    """Get names of experiments."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters()
    if param_dict is None:
        print 'No remaining experiments to run.'
        sys.exit(1)
    return param_dict['experiment_name']


def get_experiment_report(experiment_name):
    """Get list of tensorboard summaries."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        report = db_conn.get_report(experiment_name)
    return report


def get_parameters(experiment, log, random=False):
    """Get parameters for a given experiment."""
    config = credentials.postgresql_connection()
    param_dict = None
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters_and_reserve(
            experiment=experiment,
            random=random)
        log.info('Using parameters: %s' % json.dumps(param_dict, indent=4))
        if param_dict is not None:
            experiment_id = param_dict['_id']
        else:
            experiment_id = None
    if param_dict is None:
        raise RuntimeError('This experiment is complete.')
    return param_dict, experiment_id


def get_parameters_evaluation(experiment_name, log, random=False):
    """Get parameters for a given experiment."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters_for_evaluation(
            experiment_name=experiment_name,
            random=random)
        log.info('Using parameters: %s' % json.dumps(param_dict, indent=4))
        if param_dict is not None:
            experiment_id = param_dict['_id']
        else:
            experiment_id = None
    if param_dict is None:
        raise RuntimeError('This experiment is complete.')
    return param_dict, experiment_id


def initialize_database():
    """Initialize and recreate the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db()
        db_conn.return_status('CREATE')


def reset_in_process():
    """Reset the in_process table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_in_process()
    print 'Cleared the in_process table.'


def list_experiments():
    """List all experiments in the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        experiments = db_conn.list_experiments()
    return experiments


def update_performance(
        experiment_id,
        experiment,
        train_score,
        train_loss,
        val_score,
        val_loss,
        step,
        num_params,
        ckpt_path,
        results_path,
        summary_path):
    """Update performance table for an experiment."""
    config = credentials.postgresql_connection()
    perf_dict = {
        'experiment_id': experiment_id,
        'experiment': experiment,
        'train_score': train_score,
        'train_loss': train_loss,
        'val_score': val_score,
        'val_loss': val_loss,
        'step': step,
        'num_params': num_params,
        'ckpt_path': ckpt_path,
        'results_path': results_path,
        'summary_path': summary_path,
    }
    with db(config) as db_conn:
        db_conn.update_performance(perf_dict)


def get_performance(experiment_name, force_fwd=False):
    """Get performance for an experiment."""
    config = credentials.postgresql_connection()
    if force_fwd:
        config.db_ssh_forward = True
    with db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
    return perf


def main(
        initialize_db,
        reset_process=False):
    """Test the DB."""
    if reset_process:
        reset_in_process()
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_process",
        dest="reset_process",
        action='store_true',
        help='Reset the in_process table.')
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
