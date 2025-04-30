import { useEffect, useState } from 'react';
import { initializeApp } from 'firebase/app';
import {
  getFirestore, collection, query, orderBy, limit, getDocs, QueryDocumentSnapshot,
} from 'firebase/firestore';

// Initialize Firebase
const firebaseApp = initializeApp({
  apiKey: 'AIzaSyDUYjlufA4YOu6Pi3j1CI4E0_C6QOq5QW8',
  authDomain: 'jdft-colin.firebaseapp.com',
  projectId: 'jdft-colin',
});

export const db = getFirestore(firebaseApp);

// Define a TypeScript interface to describe the shape of a job
interface Job {
  id: string;
  status: string;
  last_modified_date: string;
}

export default function FirestoreData(): JSX.Element {
  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect(() => {
    const fetchJobs = async () => {
      const jobsCol = collection(db, 'jdftx-slab');
      const jobsQuery = query(jobsCol, orderBy('last_modified_epoch', 'desc'), limit(20));
      const jobSnapshot = await getDocs(jobsQuery);
      const jobsList: Job[] = jobSnapshot.docs.map((doc: QueryDocumentSnapshot) => doc.data() as Job);
      setJobs(jobsList);
    };

    fetchJobs();
  }, []);

  return (
    <table>
      <thead>
        <tr>
          <th>Id</th>
          <th>Status</th>
          <th>Last Modified</th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((job) => (
          <tr key={job.id}>
            <td>{job.id}</td>
            <td>{job.status}</td>
            <td>{job.last_modified_date}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
