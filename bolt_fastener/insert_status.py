from typing import Dict, Tuple, List, Literal, Optional
STATUS_LITERAL = Literal['IDLE', 'APPROACH', 'INSERT', 'FASTEN', 'RETRACT', 'FULL_RETRACT', 'COMPLETED']
class InsertStatus:
    """Class to manage the insertion process states and parameters"""
    
    def __init__(self,
                 approach: Optional[Tuple[float, float, float]] = None,
                 insert: Optional[Tuple[float, float, float]] = None,
                 fasten: Optional[Tuple[float, float, float]] = None,
                 retract: Optional[Tuple[float, float, float]] = None,
                 full_retract: Optional[Tuple] = None) -> None:
        self._status_list: List[STATUS_LITERAL] = ['IDLE', 'APPROACH', 'INSERT', 'FASTEN', 'RETRACT', 'FULL_RETRACT', 'COMPLETED']
        self._status: STATUS_LITERAL = 'IDLE'
        self._status_params: Dict[STATUS_LITERAL, Tuple[float, float, float] | Tuple] = {
            'APPROACH': (0, 0.01, 0.09) if approach is None else approach,
            'INSERT': (0.04, 0, 0) if insert is None else insert,
            'FASTEN': (0, 0, 0) if fasten is None else fasten,
            'RETRACT': (-0.04, 0, 0) if retract is None else retract,
            'FULL_RETRACT': (30, 15, 40, -30, 0, 50) if full_retract is None else full_retract,
        }
    
    @property
    def status(self) -> STATUS_LITERAL:
        """Get current status"""
        return self._status
    
    @status.setter
    def status(self, value: STATUS_LITERAL) -> None:
        """Set current status"""
        if value not in self._status_list:
            raise ValueError(f"Invalid status: {value}. Must be one of {self._status_list}")
        self._status = value
    
    def get_next_status(self) -> STATUS_LITERAL:
        """Get the next status in the sequence"""
        current_idx = self._status_list.index(self._status)
        if current_idx + 1 < len(self._status_list):
            return self._status_list[current_idx + 1]
        return self._status
    
    def get_current_params(self) -> Tuple[float, float, float]:
        """Get parameters for current status"""
        return self._status_params.get(self._status, (0, 0, 0))
    
    def reset(self) -> None:
        """Reset status to IDLE"""
        self._status = 'IDLE'
    
    def is_completed(self) -> bool:
        """Check if insertion process is completed"""
        return self._status == 'COMPLETED'
    
    def is_idle(self) -> bool:
        """Check if status is IDLE"""
        return self._status == 'IDLE' 